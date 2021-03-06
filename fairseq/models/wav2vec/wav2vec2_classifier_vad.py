# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import time
import sys
import contextlib
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from fairseq import checkpoint_utils, tasks, utils
from fairseq.file_io import PathManager
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer
from fairseq.modules import Fp32GroupNorm, Fp32LayerNorm, GradMultiply, GumbelVectorQuantizer, LayerNorm, \
    MultiheadAttention, SamePad, TransposeLast


def add_common_args(parser):
    parser.add_argument("--w2v-path", help="path to wav2vec 2.0 model")
    parser.add_argument("--w2v-path2", default=None, help="path2 to wav2vec 2.0 model")
    parser.add_argument("--vad-path", default=None, help="path2 to wav2vec 2.0 vad model")
    parser.add_argument("--merge-path", default=None, help="path to wav2vec 2.0 merge vad model")
    parser.add_argument(
        "--no-pretrained-weights",
        action="store_true",
        help="if true, does not load pretrained weights",
    )
    parser.add_argument(
        "--dropout-input",
        type=float,
        metavar="D",
        help="dropout to apply to the input (after feat extr)",
    )
    parser.add_argument(
        "--final-dropout",
        type=float,
        metavar="D",
        help="dropout after transformer and before final projection",
    )
    parser.add_argument(
        "--apply-mask", action="store_true", help="apply masking during fine-tuning"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        metavar="D",
        help="dropout probability inside wav2vec 2.0 model",
    )
    parser.add_argument(
        "--attention-dropout",
        type=float,
        metavar="D",
        help="dropout probability for attention weights inside wav2vec 2.0 model",
    )
    parser.add_argument(
        "--activation-dropout",
        "--relu-dropout",
        type=float,
        metavar="D",
        help="dropout probability after activation in FFN inside wav2vec 2.0 model",
    )

    parser.add_argument(
        "--mask-length", type=int, help="repeat the mask indices multiple times"
    )

    parser.add_argument(
        "--mask-prob", type=float, help="probability of replacing a token with mask"
    )

    parser.add_argument(
        "--mask-selection",
        type=str,
        choices=["static", "uniform", "normal", "poisson"],
        help="how to choose masks",
    )

    parser.add_argument(
        "--mask-other",
        type=float,
        help="stdev of the mask length in case of 'normal' selection strategy",
    )

    parser.add_argument(
        "--no-mask-overlap",
        action="store_true",
        help="whether to allow masks to overlap",
    )

    parser.add_argument(
        "--mask-channel-length", type=int, help="repeat the mask indices multiple times"
    )

    parser.add_argument(
        "--mask-channel-prob",
        type=float,
        help="probability of replacing a token with mask",
    )

    parser.add_argument(
        "--mask-channel-selection",
        type=str,
        choices=["static", "uniform", "normal", "poisson"],
        help="how to choose masks",
    )

    parser.add_argument(
        "--mask-channel-other",
        type=float,
        help="stdev of the mask length in case of 'normal' selection strategy",
    )

    parser.add_argument(
        "--no-mask-channel-overlap",
        action="store_true",
        help="whether to allow masks to overlap",
    )

    parser.add_argument(
        "--freeze-finetune-updates",
        default=0,
        type=int,
        help="dont finetune wav2vec for this many updates",
    )

    parser.add_argument(
        "--feature-grad-mult",
        default=None,
        type=float,
        help="reset feature grad mult in wav2vec 2.0 to this",
    )

    parser.add_argument(
        "--layerdrop",
        default=0.0,
        type=float,
        help="probability of dropping a layer in wav2vec 2.0",
    )

def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs,_use_new_zipfile_serialization=False)
        except Exception:
            if i == 2:
                logger.error(traceback.format_exc())

@register_model("wav2vec_class_vad")
class Wav2VecCtc_vad(BaseFairseqModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        add_common_args(parser)

    def __init__(self, w2v_encoder, args):
        super().__init__()
        self.w2v_encoder = w2v_encoder
        self.args = args

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        w2v_encoder = Wav2VecEncoder(args, task.target_dictionary)
        if args.vad_path:
            with torch.no_grad():
                arg_overrides = {
                    "dropout": args.dropout,
                    "activation_dropout": args.activation_dropout,
                    "dropout_input": args.dropout_input,
                    "attention_dropout": args.attention_dropout,
                    "mask_length": args.mask_length,
                    "mask_prob": args.mask_prob,
                    "mask_selection": args.mask_selection,
                    "mask_other": args.mask_other,
                    "no_mask_overlap": args.no_mask_overlap,
                    "mask_channel_length": args.mask_channel_length,
                    "mask_channel_prob": args.mask_channel_prob,
                    "mask_channel_selection": args.mask_channel_selection,
                    "mask_channel_other": args.mask_channel_other,
                    "no_mask_channel_overlap": args.no_mask_channel_overlap,
                    "encoder_layerdrop": args.layerdrop,
                    "feature_grad_mult": args.feature_grad_mult,
                }
                model_state_dict = checkpoint_utils.load_checkpoint_to_cpu(
                    args.vad_path, arg_overrides
                )
                new_model = cls(w2v_encoder, args).state_dict()
                model_state_dict['model'] = new_model
                state_dict = utils.move_to_cpu(model_state_dict)
                torch.save(state_dict,args.merge_path, _use_new_zipfile_serialization=False)
                #with PathManager.open(args.merge_path, "wb") as f:
                #    torch_persistent_save(state_dict, f)
            sys.exit()
        return cls(w2v_encoder, args)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_seq_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_seq_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x



class Wav2VecEncoder(FairseqEncoder):
    def __init__(self, args, tgt_dict=None):
        self.apply_mask = args.apply_mask

        arg_overrides = {
            "dropout": args.dropout,
            "activation_dropout": args.activation_dropout,
            "dropout_input": args.dropout_input,
            "attention_dropout": args.attention_dropout,
            "mask_length": args.mask_length,
            "mask_prob": args.mask_prob,
            "mask_selection": args.mask_selection,
            "mask_other": args.mask_other,
            "no_mask_overlap": args.no_mask_overlap,
            "mask_channel_length": args.mask_channel_length,
            "mask_channel_prob": args.mask_channel_prob,
            "mask_channel_selection": args.mask_channel_selection,
            "mask_channel_other": args.mask_channel_other,
            "no_mask_channel_overlap": args.no_mask_channel_overlap,
            "encoder_layerdrop": args.layerdrop,
            "feature_grad_mult": args.feature_grad_mult,
        }

        if getattr(args, "w2v_args", None) is None:
            if not args.w2v_path2 is None:
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    args.w2v_path2, arg_overrides
                )
                w2v_args = state["args"]

                #torch.save(state, args.w2v_path2, _use_new_zipfile_serialization=False)
            else:
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    args.w2v_path, arg_overrides
                )
                w2v_args = state["args"]

            if not args.w2v_path is None:
                state_base = checkpoint_utils.load_checkpoint_to_cpu(
                    args.w2v_path, arg_overrides
                )
                #torch.save(state_base, args.w2v_path, _use_new_zipfile_serialization=False)
                w2v_args_base = state_base["args"]
                if getattr(args, "encoder_attention_heads", None) is not None:
                    w2v_args_base.encoder_attention_heads = args.encoder_attention_heads

                task = tasks.setup_task(w2v_args_base)
                model_base = task.build_model(w2v_args_base)
                model_base.remove_pretraining_modules()

        else:
            state = None
            w2v_args = args.w2v_args

        self.SPK_idx = tgt_dict.symbols.index('#S')
        self.NOSPK_idx = tgt_dict.symbols.index('#NS')
        assert self.NOSPK_idx + 1 == len(tgt_dict), 'vad dataset have to be the end of datasets'
        tgt_asr_symbols = tgt_dict.symbols[:self.SPK_idx]
        tgt_vad_symbols = tgt_dict.symbols[self.SPK_idx:]

        assert args.normalize == w2v_args.normalize, 'Fine-tuning works best when data normalization is the same'

        w2v_args.data = args.data
        task = tasks.setup_task(w2v_args)
        model = task.build_model(w2v_args)

        if not args.w2v_path is None:
            model.w2v_encoder.w2v_model = model_base

        if not args.w2v_path2 is None:
            d = model.w2v_encoder.w2v_model.args.encoder_embed_dim
            model.w2v_encoder.proj = Linear(d, len(tgt_asr_symbols))
        else:
            d = w2v_args.encoder_embed_dim

        if state is not None and not args.no_pretrained_weights:
            if not args.w2v_path2 is None:
                strict = False
            else:
                strict = True
            model.load_state_dict(state["model"], strict=strict)

        if args.w2v_path2 is None:
            model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)

        if not args.w2v_path2 is None:
            self.w2v_model = model.w2v_encoder.w2v_model
            self.proj_asr = model.w2v_encoder.proj

        else:
            self.w2v_model = model

        self.final_dropout = nn.Dropout(args.final_dropout)
        self.freeze_finetune_updates = args.freeze_finetune_updates
        self.num_updates = 0

        self.vad_encoder = VAD_Encoder(
            encoder_embed_dim=512,
            ffn_embedding_dim=1024,
            num_attention_heads=4,
            conv_pos=48,  # 128 #48 #48
            conv_pos_groups=4,  # 16  #4 #16
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,
        )

        if tgt_dict is not None:
            self.post_scd_layer = Linear(512, 128)
            self.proj = Linear(128, len(tgt_vad_symbols)) # by mli test_cnn
            #self.proj = Linear(d, len(tgt_vad_symbols))
            #self.proj_asr = Linear(d, len(tgt_asr_symbols))


        else:
            self.proj = None
            self.proj_asr = None

        if args.vad_path is not None:
            state_vad = checkpoint_utils.load_checkpoint_to_cpu(
                args.vad_path, arg_overrides
            )

            vad_params = state_vad["model"]



            # load vad_encoder params
            model_dict = self.vad_encoder.state_dict()
            pretrained_dict = {k.replace('w2v_encoder.vad_encoder.', ''): v for k, v in vad_params.items() if
                               k.replace('w2v_encoder.vad_encoder.', '') in model_dict}
            model_dict.update(pretrained_dict)
            self.vad_encoder.load_state_dict(model_dict, strict=True)

            # load post_scd_layer params
            model_dict = self.post_scd_layer.state_dict()
            pretrained_dict = {k.replace('w2v_encoder.post_scd_layer.', ''): v for k, v in vad_params.items() if
                               k.replace('w2v_encoder.post_scd_layer.', '') in model_dict}
            model_dict.update(pretrained_dict)
            self.post_scd_layer.load_state_dict(model_dict, strict=True)


            #load proj_layer params
            model_dict = self.proj.state_dict()
            pretrained_dict = {k.replace('w2v_encoder.proj.',''): v for k, v in vad_params.items() if k.replace('w2v_encoder.proj.','') in model_dict}
            model_dict.update(pretrained_dict)
            self.proj.load_state_dict(model_dict, strict=True)


    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


    def forward(self, source, padding_mask, tbc=False, stage=None, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        #'''
        if not self.training:
            #stage = 'cnn_only'
            with torch.no_grad():
                if stage == 'cnn_only':
                    _, x_seq, padding_mask = self.w2v_model.extract_features(**w2v_args, vad=True, stage='cnn_only')
                    cnn_features = x_seq
                    x_seq = self.vad_encoder(cnn_features)
                    x_seq = self.post_scd_layer(x_seq)
                    if self.proj:
                        x_seq = self.proj(x_seq)
                    return {
                        "encoder_seq_out": x_seq,  # B x T x C
                        "cnn_features":cnn_features,
                        "encoder_padding_mask": padding_mask,  # B x T
                        "padding_mask": padding_mask,
                    }

                elif stage == 'no_cnn':
                    x = self.w2v_model.extract_features_no_cnn(source)
                    x_tbc = x.transpose(0, 1)
                    x_tbc = self.final_dropout(x_tbc)
                    x_tbc = self.proj_asr(x_tbc)
                    return {
                        "encoder_out": x_tbc,  # T x B x C
                    }

                else:
                    x, padding_mask = self.w2v_model.extract_features(**w2v_args)
                    x_tbc = x.transpose(0, 1)
                    x_tbc = self.final_dropout(x_tbc)

                    if self.proj:
                        with torch.no_grad():
                            x_tbc = self.proj_asr(x_tbc)
                    return {
                        "encoder_out": x_tbc,  # T x B x C
                        "encoder_padding_mask": padding_mask,  # B x T
                        "padding_mask": padding_mask,
                    }

        else:
            with torch.no_grad() if not ft else contextlib.ExitStack():
                _, x_seq, padding_mask = self.w2v_model.extract_features(**w2v_args, vad=True, stage='cnn_only')
                x_seq = self.vad_encoder(x_seq)
                x_seq = self.post_scd_layer(x_seq)
            if self.proj:
                x_seq = self.proj(x_seq)
            return {
                "encoder_seq_out": x_seq,  # B x T x C
                "encoder_padding_mask": padding_mask,  # B x T
                "padding_mask": padding_mask,
            }
        #'''

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class VAD_Encoder(nn.Module):
    def __init__(self,
                 dropout: float = 0.1,
                 encoder_embed_dim: float = 512,
                 ffn_embedding_dim: float = 1024,
                 num_attention_heads: float = 4,
                 attention_dropout: float = 0.1,
                 activation_dropout: float = 0.1,
                 activation_fn: str = "relu",
                 layer_norm_first: bool = False,
                 conv_pos: float = 128,
                 conv_pos_groups: float = 16,
                 ):
        super().__init__()

        self.dropout = dropout
        self.embedding_dim = encoder_embed_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = activation_fn

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=conv_pos,
            padding=conv_pos // 2,
            groups=conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(conv_pos), nn.GELU())
        self.layer_norm_first = layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)

        #self.apply(init_bert_params)

    def forward(self, x, padding_mask=None):
        x = self.extract_features(x, padding_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(self, x, padding_mask=None):
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x += x_conv
        #here
        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

@register_model_architecture("wav2vec_class_vad", "wav2vec_class_vad")
def base_architecture(args):
    args.no_pretrained_weights = getattr(args, "no_pretrained_weights", False)
    args.dropout_input = getattr(args, "dropout_input", 0)
    args.final_dropout = getattr(args, "final_dropout", 0)
    args.apply_mask = getattr(args, "apply_mask", False)
    args.dropout = getattr(args, "dropout", 0)
    args.attention_dropout = getattr(args, "attention_dropout", 0)
    args.activation_dropout = getattr(args, "activation_dropout", 0)

    args.mask_length = getattr(args, "mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.5)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0.5)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap", False)

    args.freeze_finetune_updates = getattr(args, "freeze_finetune_updates", 0)
    args.feature_grad_mult = getattr(args, "feature_grad_mult", 0)
    args.layerdrop = getattr(args, "layerdrop", 0.0)