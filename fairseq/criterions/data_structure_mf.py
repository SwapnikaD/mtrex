# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
static_field = 'static'
inst_pos_field = 'inst_pos_emb'  # instruction positional embedding
op_pos_field = 'op_pos_emb'  # opcode/operand positional embedding
arch_field = 'arch_emb'
byte_fields = [f'byte{i}' for i in range(1, 5)]

maskable_fields = [static_field] + byte_fields
aux_fields = [inst_pos_field] + [op_pos_field] + [arch_field]
non_byte_fields = [static_field] + [inst_pos_field] + [op_pos_field] + [arch_field]
fields = non_byte_fields + byte_fields

byte_len = 4
full_attn = False
min_chunk_len = 20
chunk_mask_relax = 0.9
last_layer = -1
cosine_embedding_loss_margin = 0.1
code_value_loss_alpha = 10


@register_criterion('data_structure_mf')
class DataStructureMFCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.fields = fields

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='data_structure_head',
                            help='name of the classification head to use')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
                hasattr(model, 'classification_heads')
                and self.args.classification_head_name in model.classification_heads
        ), 'model must provide sentence classification head for --criterion=data_structure_head'

        real_tokens = sample['target'].ne(self.task.label_dictionary.pad() - self.task.label_dictionary.nspecial)

        assert torch.all(
            real_tokens.eq(
                sample['net_input']['src_tokens'][self.fields[-3]].ne(self.padding_idx_dict[self.fields[-3]])))

        sample_size = real_tokens.int().sum().float()

        logits, _ = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
        targets = model.get_targets(sample, [logits])[real_tokens].view(-1)

        lprobs = F.log_softmax(logits[real_tokens, :], dim=-1, dtype=torch.float32)
        loss = F.nll_loss(lprobs, targets, reduction='sum')

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }

        # print(sample['target'].size())
        preds = logits[real_tokens, :].argmax(dim=1)
        logging_output['ncorrect_total'] = utils.item((preds == targets).sum())
        logging_output['ncorrect'] = utils.item(((preds == targets) * (targets != 0)).sum())

        logging_output['ntype'] = utils.item((targets != 0).sum().item())
        logging_output['ntype_pred'] = utils.item((preds != 0).sum().item())

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get('nsentences', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect_total' in logging_outputs[0]:
            ncorrect_total = sum(log.get('ncorrect_total', 0) for log in logging_outputs)
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            ntype = sum(log.get('ntype', 0) for log in logging_outputs)
            ntype_pred = sum(log.get('ntype_pred', 0) for log in logging_outputs)

            precision = ncorrect / (ntype_pred + 1e-5)
            recall = ncorrect / (ntype + 1e-5)
            F1 = 2 * (precision * recall) / (precision + recall + 1e-5)
            metrics.log_scalar('precision', 100.0 * precision, sample_size, round=1)
            metrics.log_scalar('recall', 100.0 * recall, sample_size, round=1)
            metrics.log_scalar('F1', 100.0 * F1, sample_size, round=1)
            metrics.log_scalar('accuracy', 100.0 * ncorrect_total / sample_size, sample_size, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
