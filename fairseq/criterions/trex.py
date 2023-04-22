# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, modules, utils
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
from fairseq.criterions import FairseqCriterion, register_criterion
import random


@register_criterion("trex")
class TrexLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, task, tpu=False):
        super().__init__(task)
        self.tpu = tpu

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_code = sample["target"][static_field].ne(self.padding_idx_dict[static_field])
        masked_value = sample["target"][byte_fields[0]].ne(1)

        # Rare: when all tokens are not masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if self.tpu:
            masked_code = None  # always project all tokens on TPU
            masked_value = None  # always project all tokens on TPU
        elif masked_code.device == torch.device("cpu"):
            if not masked_code.any():
                masked_code = None
            if not masked_value.any():
                masked_value = None
        else:
            masked_code = torch.where(
                masked_code.any(),
                masked_code,
                masked_code.new([True]),
            )
            masked_value = torch.where(
                masked_value.any(),
                masked_value,
                masked_value.new([True]),
            )

        output = model(**sample["net_input"], masked_code=masked_code, masked_value=masked_value)[0]

        pred_logits_code, pred_value = output['code'], output['value']
        targets_code = sample["target"][static_field]

        if masked_code is not None:
            targets_code = targets_code[masked_code]

        if masked_value is not None:
            targets_value_stacked = torch.stack(
                [sample["target"][field][masked_value] for field in byte_fields], dim=1)

        sample_size_code = masked_code.int().sum()
        sample_size_value = masked_value.int().sum() * byte_len
        sample_size = sample_size_code + sample_size_value

        code_loss = modules.cross_entropy(
            pred_logits_code.view(-1, pred_logits_code.size(-1)),
            targets_code.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx_dict[static_field],
        )

        value_loss = F.mse_loss(
            pred_value.float(),
            targets_value_stacked.float(),
            reduction='sum'
        )

        loss = code_loss + code_value_loss_alpha * value_loss

        if random.random() < 0.001:  # only randomly log some prediction in case screen flushing
            for i, field in enumerate(byte_fields):
                print(f'{field} tgt value:', sample["target"][field][masked_value].view(-1)[5:10].tolist())
                print(f'{field} pred value:', pred_value[5:10, i].view(-1).tolist())

            targets_code_idx = targets_code.view(-1)[5:10]
            pred_code_idx = torch.argmax(pred_logits_code.view(-1, pred_logits_code.size(-1))[5:10], dim=-1)
            print(f'tgt code:', self.task.source_dictionary[static_field].string(targets_code_idx))
            print(f'pred code:', self.task.source_dictionary[static_field].string(pred_code_idx))

        logging_output = {
            "loss": loss.data,
            'code_loss': code_loss.data,
            'value_loss': value_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "sample_size_code": sample_size_code,
            "sample_size_value": sample_size_value,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        code_loss_sum = sum(log.get("code_loss", 0) for log in logging_outputs)
        value_loss_sum = sum(log.get("value_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        sample_size_code = sum(log.get("sample_size_code", 0) for log in logging_outputs)
        sample_size_value = sum(log.get("sample_size_value", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("code_loss", code_loss_sum / sample_size_code / math.log(2), sample_size_code, round=3)
        metrics.log_scalar("value_loss_mse", value_loss_sum / sample_size_value, sample_size_value, round=3)
        metrics.log_derived("code_ppl", lambda meters: utils.get_perplexity(meters["code_loss"].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
