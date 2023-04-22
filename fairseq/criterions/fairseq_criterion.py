# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
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

import inspect
from typing import Any, Dict, List

from torch.nn.modules.loss import _Loss
from fairseq import metrics, utils
from fairseq.data import Dictionary
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import gen_parser_from_dataclass


class FairseqCriterion(_Loss):
    def __init__(self, task):
        super().__init__()
        self.task = task

        if hasattr(task, "target_dictionary"):
            if type(task.target_dictionary) is Dictionary:
                tgt_dict = task.target_dictionary
                self.padding_idx = tgt_dict.pad() if tgt_dict is not None else -100
            elif type(task.target_dictionary) is dict:
                self.padding_idx_dict = {}
                for field in maskable_fields:
                    self.padding_idx_dict[field] = task.target_dictionary[field].pad()

    @classmethod
    def add_args(cls, parser):
        """Add criterion-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @classmethod
    def build_criterion(cls, cfg: FairseqDataclass, task):
        """Construct a criterion from command-line args."""
        # arguments in the __init__.
        init_args = {}
        for p in inspect.signature(cls).parameters.values():
            if (
                    p.kind == p.POSITIONAL_ONLY
                    or p.kind == p.VAR_POSITIONAL
                    or p.kind == p.VAR_KEYWORD
            ):
                # we haven't implemented inference for these argument types,
                # but PRs welcome :)
                raise NotImplementedError("{} not supported".format(p.kind))

            assert p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}

            if p.name == "task":
                init_args["task"] = task
            elif p.name == "cfg":
                init_args["cfg"] = cfg
            elif hasattr(cfg, p.name):
                init_args[p.name] = getattr(cfg, p.name)
            elif p.default != p.empty:
                pass  # we'll use the default value
            else:
                raise NotImplementedError(
                    "Unable to infer Criterion arguments, please implement "
                    "{}.build_criterion".format(cls.__name__)
                )
        return cls(**init_args)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        raise NotImplementedError

    @staticmethod
    def aggregate_logging_outputs(
            logging_outputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning(
            "The aggregate_logging_outputs API is deprecated. "
            "Please use the reduce_metrics API instead."
        )
        raise NotImplementedError

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) -> None:
        """Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning(
            "Criterions should implement the reduce_metrics API. "
            "Falling back to deprecated aggregate_logging_outputs API."
        )
        agg_logging_outputs = cls.aggregate_logging_outputs(logging_outputs)
        for k, v in agg_logging_outputs.items():
            if k in {"nsentences", "ntokens", "sample_size"}:
                continue
            metrics.log_scalar(k, v)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


class LegacyFairseqCriterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(task=task)
        self.args = args

        utils.deprecation_warning(
            "Criterions should take explicit arguments instead of an "
            "argparse.Namespace object, please update your criterion by "
            "extending FairseqCriterion instead of LegacyFairseqCriterion."
        )

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        return cls(args, task)
