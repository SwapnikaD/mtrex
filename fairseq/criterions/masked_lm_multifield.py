# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, modules, utils
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



@register_criterion('masked_lm_multifield')
class MaskedLmMultifieldLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        fields = fields
        # compute MLM loss
        masked_tokens = sample['target'][fields[0]].ne(self.padding_idx_dict[fields[0]])
        sample_size = masked_tokens.int().sum().item()

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        # if sample_size == 0:
        #     masked_tokens = None

        masked_tokens = torch.where(
            masked_tokens.any(),
            masked_tokens,
            masked_tokens.new([True]),
        )

        logits = model(**sample['net_input'], masked_tokens=masked_tokens)[0]
        targets = model.get_targets(sample, [logits])

        # Which field to predict
        output_langs = self.args.output_lang.split(',')
        trace_weight = float(self.args.trace_weight)
        for output_lang in output_langs:
            assert output_lang in logits.keys()

        loss = 0
        for field in output_langs:

            if masked_tokens is not None:
                targets[field] = targets[field][masked_tokens]

            if field == fields[0]:  # static code loss
                loss += modules.cross_entropy(
                    logits[field].view(-1, logits[field].size(-1)),
                    targets[field].view(-1),
                    reduction='sum',
                    ignore_index=self.padding_idx_dict[field],
                )
            else:
                loss += trace_weight * modules.cross_entropy(
                    logits[field].view(-1, logits[field].size(-1)),
                    targets[field].view(-1),
                    reduction='sum',
                    ignore_index=self.padding_idx_dict[field],
                )

        logging_output = {
            'loss': loss.data / len(output_langs),
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
