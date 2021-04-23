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
from fairseq.modules.transformer_sentence_encoder import init_bert_params
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


@register_model("wav2vec_class_TS_vad_v1")
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
                #torch.save(state, args.w2v_path2, _use_new_zipfile_serialization=False)
                #print('111,done', args.w2v_path2)
                #sys.exit()
                w2v_args = state["args"]
            else:
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    args.w2v_path, arg_overrides
                )
                w2v_args = state["args"]

            if not args.w2v_path is None:

                state_base = checkpoint_utils.load_checkpoint_to_cpu(
                    args.w2v_path, arg_overrides
                )
                w2v_args_base = state_base["args"]
                if getattr(args, "encoder_attention_heads", None) is not None:
                    w2v_args_base.encoder_attention_heads = args.encoder_attention_heads

                task = tasks.setup_task(w2v_args_base)
                model_base = task.build_model(w2v_args_base)
                model_base.remove_pretraining_modules()

        else:
            state = None
            w2v_args = args.w2v_args

        self.NOSPK_idx = tgt_dict.symbols.index('#NS')
        tgt_asr_symbols = tgt_dict.symbols[:self.NOSPK_idx]


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
            #d = self.w2v_model.args.encoder_embed_dim
            #print(11111,self.proj_asr.weight)
            #print(2222,state["model"]["w2v_encoder.proj.weight"])
            #sys.exit()
        else:
            #d = w2v_args.encoder_embed_dim
            self.w2v_model = model

        self.final_dropout = nn.Dropout(args.final_dropout)
        self.freeze_finetune_updates = args.freeze_finetune_updates
        self.num_updates = 0
        #'''
        self.scd_encoder = SCDTransformerEncoder(
            encoder_embed_dim=512,
            ffn_embedding_dim=3072,
            num_attention_heads=8,
            conv_pos=96, #128
            conv_pos_groups=16, #16
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,
            num_layers=6,
        )
        #'''
        #self.pre_scd_layer = Linear(512+512, 512) #128
        self.scd_classifier = SCD_classifier(
            input_dim = 1024,
            hidden1_dim = 512,
            hidden2_dim = 128,
            activation_fn = "relu",)
        #self.post_scd_layer = Linear(512, 128)  # 128

        if tgt_dict is not None:
            self.proj = Linear(128, 3) # by mli test_cnn



        else:
            self.proj = None
            self.proj_asr = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


    def forward(self, source, padding_mask,input_embeddings=None, tbc=False, stage=None, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates
        '''
        with torch.no_grad() if not ft else contextlib.ExitStack():
            _, x_seq, padding_mask = self.w2v_model.extract_features(**w2v_args, vad=True, stage='cnn_only')
        if self.proj:
            x_seq = self.proj(x_seq)
        return {
            "encoder_seq_out": x_seq,  # B x T x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
            }
        '''

        #'''
        if not self.training:
            #stage = 'cnn_only'
            with torch.no_grad():
                if stage=='cnn_only':
                    _, x_seq, padding_mask = self.w2v_model.extract_features(**w2v_args, vad=True, stage='cnn_only')
                    ts_embedding = input_embeddings.expand(input_embeddings.size(0),x_seq.size(1),input_embeddings.size(-1))

                    features = self.scd_encoder(x_seq)  # ,_
                    x_seq = torch.cat((features, ts_embedding), dim=-1)
                    x_seq = self.scd_classifier(x_seq)

                    if self.proj:
                        x_seq = self.proj(x_seq)

                    return {
                        "encoder_seq_out": x_seq,  # B x T x C
                        "features": features,
                        "ts_embedding": ts_embedding,
                        "encoder_padding_mask": padding_mask,  # B x T
                        "padding_mask": padding_mask,
                    }

                else:
                    x,  padding_mask = self.w2v_model.extract_features(**w2v_args)#, vad=True,stage='cnn_vad_asr')
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

            ts_embedding = input_embeddings.expand(input_embeddings.size(0),x_seq.size(1),input_embeddings.size(-1))
            features = self.scd_encoder(x_seq) #,_
            x_seq = torch.cat((features, ts_embedding), dim=-1)
            x_seq = self.scd_classifier(x_seq)

            if self.proj:
                x_seq = self.proj(x_seq)

            return {
                "encoder_seq_out": x_seq,  # B x T x C
                "features": features,
                "ts_embedding": ts_embedding,
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

class SCDTransformerEncoder(nn.Module):
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
                 num_layers = 6
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
        # modified by mli

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=self.ffn_embedding_dim,
                num_attention_heads=self.num_attention_heads,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout,
                activation_dropout=self.activation_dropout,
                )
                for _ in range(num_layers)
            ]
            )

        self.layer_norm_first = layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None):
        x = self.extract_features(x, padding_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(self, x, padding_mask=None):
        #print(1111,x.size())
        x_conv = self.pos_conv(x.transpose(1, 2))
        #print(2222,x_conv.size())
        #sys.exit()
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv
        #here
        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        for i, layer in enumerate(self.layers):
            x, z = layer(x)
        #x,_ = self.scd_layer(x)
        x = x.transpose(0, 1)
        return x

class SCD_classifier(nn.Module):
    def __init__(
        self,
        input_dim: float = 1024,
        hidden1_dim: float = 512,
        hidden2_dim: float = 128,
        activation_fn: str = "relu",
    ) -> None:
        super().__init__()
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.fc1 = Linear(input_dim, hidden1_dim)
        self.fc2 = Linear(hidden1_dim, hidden2_dim)

    def forward(
            self,
            x: torch.Tensor,
    ):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        return x

class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:
        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        #print(12312312,embedding_dim,ffn_embedding_dim,num_attention_heads)
        #sys.exit()

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)


    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            #print(43254234)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn



@register_model_architecture("wav2vec_class_TS_vad_v1", "wav2vec_class_TS_vad_v1")
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