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


@register_model("wav2vec_class_vad_scd")
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
                task = tasks.setup_task(w2v_args_base)
                model_base = task.build_model(w2v_args_base)
                model_base.remove_pretraining_modules()

        else:
            state = None
            w2v_args = args.w2v_args

        self.SPK_idx = tgt_dict.symbols.index('#S')
        self.NOSPK_idx = tgt_dict.symbols.index('#NS')
        self.SC_idx = tgt_dict.symbols.index('#SC')
        #print(13123,self.SPK_idx,self.NOSPK_idx,self.SC_idx,tgt_dict.symbols)

        assert self.SC_idx + 1 == len(tgt_dict), 'vad dataset have to be the end of datasets'
        tgt_asr_symbols = tgt_dict.symbols[:self.SPK_idx]
        tgt_scd_symbols = tgt_dict.symbols[self.SPK_idx:]


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
        '''
        self.scd_layer = TransformerSentenceEncoderLayer(
            embedding_dim=512,
            ffn_embedding_dim=1024,
            num_attention_heads=4,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,
        )
        '''
        self.pre_scd_layer = Linear(512, 256) #128
        self.scd_layer = RNN(256,
        unit="GRU",  #GRU
        hidden_size=128,  #32
        num_layers=2, #1
        bias=True,
        dropout=0,
        bidirectional=False,
        concatenate=False,
        pool=None,)
        self.post_scd_layer = Linear(128, 128)  # 128

        if tgt_dict is not None:
            self.proj = Linear(128, len(tgt_scd_symbols)) # by mli test_cnn
            #self.proj = Linear(d, len(tgt_scd_symbols))
            #self.proj_asr = Linear(d, len(tgt_asr_symbols))


        else:
            self.proj = None
            self.proj_asr = None

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
                    x_seq = self.scd_layer(self.pre_scd_layer(x_seq)) #, _
                    x_seq = self.post_scd_layer(x_seq)
                    if self.proj:
                        x_seq = self.proj(x_seq)
                    return {
                        "encoder_seq_out": x_seq,  # B x T x C
                        "encoder_padding_mask": padding_mask,  # B x T
                        "padding_mask": padding_mask,
                    }
                else:
                    x, x_seq, padding_mask = self.w2v_model.extract_features(**w2v_args, vad=True,stage='cnn_vad_asr')
                    x_tbc = x.transpose(0, 1)
                    # time average
                    x = torch.mean(x, dim=1, keepdim=False)
                    x = self.final_dropout(x)
                    x_tbc = self.final_dropout(x_tbc)
                    feats = x
                    # modified by mli
                    x_seq = self.scd_layer(self.pre_scd_layer(x_seq)) #, _
                    x_seq = self.post_scd_layer(x_seq)
                    if self.proj:
                        x_seq = self.proj(x_seq)
                        with torch.no_grad():
                            x_tbc = self.proj_asr(x_tbc)
                    return {
                        "encoder_out": x_tbc,  # T x B x C
                        "encoder_seq_out": x_seq,  # B x T x C
                        "features_out": feats,
                        "encoder_padding_mask": padding_mask,  # B x T
                        "padding_mask": padding_mask,
                        }

        else:
            with torch.no_grad() if not ft else contextlib.ExitStack():
                _, x_seq, padding_mask = self.w2v_model.extract_features(**w2v_args, vad=True, stage='cnn_only')
            #print(111111,x_seq.size())
            x_seq = self.scd_layer(self.pre_scd_layer(x_seq)) #,_
            #print(2222,x_seq.size())
            x_seq = self.post_scd_layer(x_seq)
            #print(33333,x_seq.size())
            if self.proj:
                x_seq = self.proj(x_seq)
            #print(4444,x_seq.size())
            #sys.exit()
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

class RNN(nn.Module):
    """Recurrent layers
    Parameters
    ----------
    n_features : `int`
        Input feature shape.
    unit : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    hidden_size : `int`, optional
        Number of features in the hidden state h. Defaults to 16.
    num_layers : `int`, optional
        Number of recurrent layers. Defaults to 1.
    bias : `boolean`, optional
        If False, then the layer does not use bias weights. Defaults to True.
    dropout : `float`, optional
        If non-zero, introduces a Dropout layer on the outputs of each layer
        except the last layer, with dropout probability equal to dropout.
        Defaults to 0.
    bidirectional : `boolean`, optional
        If True, becomes a bidirectional RNN. Defaults to False.
    concatenate : `boolean`, optional
        Concatenate output of each layer instead of using only the last one
        (which is the default behavior).
    pool : {'sum', 'max', 'last', 'x-vector'}, optional
        Temporal pooling strategy. Defaults to no pooling.
    """

    def __init__(
        self,
        n_features,
        unit="LSTM",
        hidden_size=16,
        num_layers=1,
        bias=True,
        dropout=0,
        bidirectional=False,
        concatenate=False,
        pool=None,
    ):
        super().__init__()

        self.n_features = n_features

        self.unit = unit
        Klass = getattr(nn, self.unit)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.concatenate = concatenate
        self.pool = pool
        self.pool_ = TemporalPooling.create(pool) if pool is not None else None

        if num_layers < 1:
            msg = '"bidirectional" must be set to False when num_layers < 1'
            if bidirectional:
                raise ValueError(msg)
            msg = '"concatenate" must be set to False when num_layers < 1'
            if concatenate:
                raise ValueError(msg)
            return

        if self.concatenate:

            self.rnn_ = nn.ModuleList([])
            for i in range(self.num_layers):

                if i > 0:
                    input_dim = self.hidden_size
                    if self.bidirectional:
                        input_dim *= 2
                else:
                    input_dim = self.n_features

                if i + 1 == self.num_layers:
                    dropout = 0
                else:
                    dropout = self.dropout

                rnn = Klass(
                    input_dim,
                    self.hidden_size,
                    num_layers=1,
                    bias=self.bias,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=self.bidirectional,
                )

                self.rnn_.append(rnn)

        else:
            self.rnn_ = Klass(
                self.n_features,
                self.hidden_size,
                num_layers=self.num_layers,
                bias=self.bias,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )

    def forward(self, features, return_intermediate=False):
        """Apply recurrent layer (and optional temporal pooling)
        Parameters
        ----------
        features : `torch.Tensor`
            Features shaped as (batch_size, n_frames, n_features)
        return_intermediate : `boolean`, optional
            Return intermediate RNN hidden state.
        Returns
        -------
        output : `torch.Tensor`
            TODO. Shape depends on parameters...
        intermediate : `torch.Tensor`
            (num_layers, batch_size, hidden_size * num_directions)
        """

        if self.num_layers < 1:

            if return_intermediate:
                msg = (
                    '"return_intermediate" must be set to False ' "when num_layers < 1"
                )
                raise ValueError(msg)

            output = features

        else:

            if return_intermediate:
                num_directions = 2 if self.bidirectional else 1

            if self.concatenate:

                if return_intermediate:
                    msg = (
                        '"return_intermediate" is not supported '
                        'when "concatenate" is True'
                    )
                    raise NotADirectoryError(msg)

                outputs = []

                hidden = None
                output = None
                # apply each layer separately...

                for i, rnn in enumerate(self.rnn_):
                    if i > 0:
                        output, hidden = rnn(output, hidden)
                    else:
                        output, hidden = rnn(features)
                    outputs.append(output)

                # ... and concatenate their output
                output = torch.cat(outputs, dim=2)

            else:
                output, hidden = self.rnn_(features)

                if return_intermediate:
                    if self.unit == "LSTM":
                        h = hidden[0]
                    elif self.unit == "GRU":
                        h = hidden

                    # to (num_layers, batch_size, num_directions * hidden_size)
                    h = h.view(self.num_layers, num_directions, -1, self.hidden_size)
                    intermediate = (
                        h.transpose(2, 1)
                        .contiguous()
                        .view(self.num_layers, -1, num_directions * self.hidden_size)
                    )

        if self.pool_ is not None:
            output = self.pool_(output)

        if return_intermediate:
            return output, intermediate

        return output

    def dimension():
        doc = "Output features dimension."

        def fget(self):
            if self.num_layers < 1:
                dimension = self.n_features
            else:
                dimension = self.hidden_size

            if self.bidirectional:
                dimension *= 2

            if self.concatenate:
                dimension *= self.num_layers

            if self.pool == "x-vector":
                dimension *= 2

            return dimension

        return locals()

    dimension = property(**dimension())

    def intermediate_dimension(self, layer):
        if self.num_layers < 1:
            dimension = self.n_features
        else:
            dimension = self.hidden_size

        if self.bidirectional:
            dimension *= 2

        return dimension

@register_model_architecture("wav2vec_class_vad_scd", "wav2vec_class_vad_scd")
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