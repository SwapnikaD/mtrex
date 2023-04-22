# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules.trex_encoder import DEFAULT_MIN_PARAMS_TO_WRAP, TrexEncoder
from .hub_interface import TrexHubInterface

logger = logging.getLogger(__name__)


@register_model("trex")
class TrexModel(FairseqEncoderModel):
    @classmethod
    def hub_models(cls):
        return {
            "trex.base": "",
        }

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

    def similarity_pair(self, concat_in: torch.Tensor):
        return self.classification_heads.similarity_pair(concat_in)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--load-checkpoint-heads",
            action="store_true",
            help="(re-)register and load heads when loading checkpoints",
        )
        parser.add_argument(
            "--untie-weights-trex",
            action="store_true",
            help="Untie weights between embeddings and classifiers in Trex",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument(
            "--quant-noise-pq",
            type=float,
            metavar="D",
            default=0,
            help="iterative PQ quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-pq-block-size",
            type=int,
            metavar="D",
            default=8,
            help="block size of quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-scalar",
            type=float,
            metavar="D",
            default=0,
            help="scalar quantization noise and scalar quantization at training time",
        )
        # args for "Better Fine-Tuning by Reducing Representational Collapse" (Aghajanyan et al. 2020)
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the classification head",
        )
        # args for Fully Sharded Data Parallel (FSDP) training
        parser.add_argument(
            "--min-params-to-wrap",
            type=int,
            metavar="D",
            default=DEFAULT_MIN_PARAMS_TO_WRAP,
            help=(
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            )
        )

        # byte combine
        parser.add_argument(
            '--seq-combine',
            default='sum',
            choices=['sum', 'concat'],
            help='strategies to combine the embeddings of 5 input sequences, default is sum')

        # byte combine
        parser.add_argument(
            '--input-combine',
            default='cnn',
            choices=['cnn', 'sum'],
            help='strategies to combine the byte value input embeddings, default is char-level cnn (with highway networks)')

        # drop fields
        parser.add_argument(
            '--drop-field',
            default='none',
            choices=['none', 'static', 'inst_pos_emb', 'op_pos_emb', 'arch_emb', 'bytes'],
            help='ablation study only. drop selected input sequence')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = Encoder(args, task.source_dictionary)
        return cls(args, encoder)

    def forward(
            self,
            src_tokens: Dict[str, torch.Tensor],
            src_lengths: Optional[torch.Tensor] = None,
            features_only: bool = False,
            return_all_hiddens: bool = False,
            classification_head_name: Optional[str] = None,
            masked_code: Optional[torch.Tensor] = None,
            masked_value: Optional[torch.Tensor] = None
    ):
        if classification_head_name is not None:
            features_only = True

        x, extra = self.encoder(src_tokens, src_lengths, features_only, return_all_hiddens, masked_code, masked_value)

        # hacky workaround to deal with torchscript 'unable to extract string literal'
        if classification_head_name is not None:
            x = {
                'features': self.classification_heads.similarity(x['features'])
            }
        return x, extra

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def register_similarity_head(
            self, name, num_classes=None, inner_dim=None
    ):
        """Register a embedding transformation head for computing cosine similarity."""
        if name in self.classification_heads:
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if inner_dim != prev_inner_dim:
                logger.warning(f're-registering head {name} with inner_dim {inner_dim} (prev: {prev_inner_dim})')

        self.classification_heads[name] = TrexSimilarityHead(
            # input_dim=self.args.encoder_embed_dim * 3,
            input_dim=self.args.encoder_embed_dim,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
            do_spectral_norm=self.args.spectral_norm_classification_head,
        )

    def register_similarity_pair_head(
            self, name, num_classes=None, inner_dim=None
    ):
        """Register a head for computing similarity of two embeddings."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with out_dim {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = TrexEmbeddingPairHead(
            input_dim=self.args.encoder_embed_dim * 4,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            output_dim=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            do_spectral_norm=self.args.spectral_norm_classification_head,
        )

    # @property
    # def supported_targets(self):
    #     return {"self"}

    @classmethod
    def from_pretrained(
            cls,
            model_name_or_path,
            checkpoint_file="model.pt",
            data_name_or_path=".",
            bpe="gpt2",
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True
        )

        logger.info(x["args"])
        return TrexHubInterface(x["args"], x["task"], x["models"][0])

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + "decoder"):
                new_k = prefix + "encoder" + k[len(prefix + "decoder"):]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # rename emb_layer_norm -> layernorm_embedding
        for k in list(state_dict.keys()):
            if ".emb_layer_norm." in k:
                new_k = k.replace(".emb_layer_norm.", ".layernorm_embedding.")
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads."):].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
                ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
                ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                        num_classes
                        != self.classification_heads[head_name].out_proj.out_features
                        or inner_dim
                        != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v


class TrexLMClassificationHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens: Optional[torch.Tensor] = None):
        # Only project the masked tokens while training,
        # saves both memory and computation
        # if masked_tokens is not None:
        features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class TrexLMValueRegressionHead(nn.Module):
    """Head for masked language modeling on trace value as a regression task."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        # predict all bytes together
        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens: Optional[torch.Tensor] = None):
        # Only project the masked tokens while training,
        # saves both memory and computation
        # if masked_tokens is not None:
        features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        # the bytes are scaled between [0,1]
        x = torch.sigmoid(x)
        return x


class TrexSimilarityHead(nn.Module):
    """Head for predicting embeddings for similarity."""

    def __init__(
            self,
            input_dim,
            activation_fn,
            pooler_dropout,
            q_noise=0,
            qn_block_size=8,
            do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, input_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(input_dim, input_dim), q_noise, qn_block_size
        )
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError(
                    "Attempting to use Spectral Normalization with Quant Noise. This is not officially supported"
                )
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features):
        # CLS + mean pooling + max pooling
        # x = torch.cat((features[:, 0, :], torch.mean(features, dim=1), torch.max(features, dim=1)[0]), dim=-1)
        x = torch.mean(features, dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = F.normalize(x)
        return x


class TrexEmbeddingPairHead(nn.Module):
    """Head for sentence-level embedding tasks (e.g., for detect embedding pair)."""

    def __init__(
            self,
            input_dim,
            inner_dim,
            output_dim,
            activation_fn,
            pooler_dropout,
            do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, output_dim)
        if do_spectral_norm:
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, concatenated_x):
        x = self.dropout(concatenated_x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Encoder(FairseqEncoder):
    """Trex encoder."""

    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        # set any missing default values
        base_architecture(args)
        self.args = args
        self.fields = fields

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        embed_tokens = nn.ModuleDict(
            {field: self.build_embedding(len(dictionary[field]), args.encoder_embed_dim, dictionary[field].pad()) for
             field in non_byte_fields})

        self.sentence_encoder = self.build_encoder(args, dictionary, embed_tokens)

        self.lm_code_head, self.lm_value_head = self.build_lm_head(
            embed_dim=args.encoder_embed_dim,
            output_dim_code=len(dictionary[static_field]),
            output_dim_value=byte_len,
            activation_fn=args.activation_fn,
            weight=(
                self.sentence_encoder.embed_tokens[static_field].weight
                if not args.untie_weights_trex
                else None
            ),
        )

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TrexEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

    def build_lm_head(self, embed_dim, output_dim_code, output_dim_value, activation_fn, weight):
        code_head = TrexLMClassificationHead(embed_dim, output_dim_code, activation_fn, weight)
        value_head = TrexLMValueRegressionHead(embed_dim, output_dim_value, activation_fn)
        return code_head, value_head

    def forward(
            self,
            src_tokens: Dict[str, torch.Tensor],
            src_lengths: Optional[torch.Tensor] = None,
            features_only: bool = False,
            return_all_hiddens: bool = False,
            masked_code: Optional[torch.Tensor] = None,
            masked_value: Optional[torch.Tensor] = None
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            src_lengths (LongTensor): input tokens length
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            masked_code (bool, optional): the masked code token from static code
            masked_value (bool, optional): the masked code token from static code

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
        if not features_only:
            x = self.output_layer(x, masked_code=masked_code, masked_value=masked_value)
        return x, extra

    def extract_features(self, src_tokens: Dict[str, torch.Tensor], return_all_hiddens: bool = False):
        encoder_out = self.sentence_encoder(
            src_tokens,
            return_all_hiddens=return_all_hiddens,
        )
        # T x B x C -> B x T x C
        features = encoder_out["encoder_out"][0].transpose(0, 1)
        inner_states = encoder_out["encoder_states"] if return_all_hiddens else None

        # for torchscript, has to be dictionary
        return {'features': features}, {"inner_states": inner_states}

    def output_layer(self,
                     features: Dict[str, torch.Tensor],
                     masked_code: Optional[torch.Tensor] = None,
                     masked_value: Optional[torch.Tensor] = None
                     ):
        return {
            'code': self.lm_code_head(features['features'], masked_code),
            'value': self.lm_value_head(features['features'], masked_value)
        }

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


@register_model_architecture("trex", "trex")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)

    args.max_source_positions = getattr(args, "max_positions", 512)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    # BERT has a few structural differences compared to the original Transformer
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.untie_weights_trex = getattr(args, "untie_weights_trex", False)

    # Adaptive input config
    args.adaptive_input = getattr(args, "adaptive_input", False)

    # LayerDrop config
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)

    # Quantization noise config
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    # R4F config
    args.spectral_norm_classification_head = getattr(
        args, "spectral_norm_classification_head", False
    )


@register_model_architecture("trex", "trex_prenorm")
def trex_prenorm_architecture(args):
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    base_architecture(args)


@register_model_architecture("trex", "trex_base")
def trex_base_architecture(args):
    base_architecture(args)


@register_model_architecture("trex", "trex_large")
def trex_large_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    base_architecture(args)


@register_model_architecture("trex", "trex_xlm")
def xlm_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1280)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1280 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    base_architecture(args)
