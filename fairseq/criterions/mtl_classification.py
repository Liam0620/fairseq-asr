# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
import torch.nn.functional as F
import torch
import torch.nn as nn

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data.data_utils import post_process

from fairseq.logging.meters import safe_round

@register_criterion('mtl_classification')
class MTL_ClassifyCriterion(FairseqCriterion):

    def __init__(self, task, wer_args=None, zero_infinity=None,sentence_avg=None,class_num=None):
        super().__init__(task)

        self.post_process = "letter"

        if wer_args is not None:
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            wer_compute_kenlm, wer_lexicon, lm_w, ws_w = eval(wer_args)

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = wer_compute_kenlm
            dec_args.lexicon = wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = lm_w
            dec_args.word_score = ws_w
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        self.zero_infinity = zero_infinity
        self.sentence_avg = sentence_avg

        if task.target_dictionary:
            num_classes = len(task.target_dictionary.symbols)
            self.blank_idx = task.target_dictionary.bos()
            self.pad_idx = task.target_dictionary.pad()
            self.eos_idx = task.target_dictionary.eos()
            self.unk_idx = task.target_dictionary.unk()
            self.voice_idx = task.target_dictionary.symbols.index('#S')

        else:
            if not class_num:
                print('please set the class_num')
                sys.exit()
            num_classes = class_num
        self.criterion_sid = AM_softmax(768, num_classes, margin=0.2, scale=30)
        self.criterion_lid = AM_softmax(768, num_classes, margin=0.2, scale=30)
        #self.criterion_vad = nn.NLLLoss(reduction='sum')
        self.criterion_vad =  nn.CrossEntropyLoss() #
        self.criterion_ctc =  nn.CTCLoss(blank=0, reduction='mean',zero_infinity=True)  #  CTCLoss()

        #print(11111,task.target_dictionary.symbols[:5])


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--zero-infinity", action="store_true", help="zero inf loss"
        )
        try:
            parser.add_argument(
                "--remove-bpe",
                "--post-process",
                default="letter",
                help="remove BPE tokens before scoring (can be set to sentencepiece, letter, and more)",
            )
        except:
            pass  # this option might have been added from eval args
        parser.add_argument(
            "--wer-args",
            type=str,
            default=None,
            help="options for wer computation on valid set using 4 gram lm. this should be a tuple of 4 elements: path to 4-gram lm, \
                path to lexicon, lm score, word score",
        )
    
    def extract_logits(self,model,sample):
        net_output = model(**sample)
        feats = net_output["features_out"]
        logists = self.criterion_lid.extract_logits(feats)
        return logists

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        #print('sample',sample)
        lid,sid,vad,asr = False,False,False,False
        assert len(set(sample['task'])) == 1
        task = sample['task'][0]
        if task=='lid':
            lid = True
        elif task=='sid':
            sid = True
        elif task=='vad':
            vad = True
        elif task=="asr":
            asr = True

        net_output = model(**sample['net_input'])
        feats = net_output["features_out"]
        logits_vad = net_output['encoder_seq_out']
        slb_len = logits_vad.view(-1, logits_vad.size(-1)).size(0)

        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            non_padding_mask = ~net_output["padding_mask"]
            input_lengths = non_padding_mask.long().sum(-1)

        pad_mask = (sample["target"] != self.pad_idx) & (
                sample["target"] != self.eos_idx
        )


        targets = model.get_targets(sample, net_output).view(-1).long()


        bsz = model.get_targets(sample, net_output).size(0)

        if lid:
            targets_slb = torch.tensor([self.voice_idx]).repeat(slb_len).cuda()
            targets_cls = targets
            lid_w, sid_w, vad_w,asr_w = 1,0,0,0
            #print(111111,targets_slb.size(),targets_cls.size())
            targets_flat = torch.tensor([self.unk_idx]).repeat(len(targets_cls))
            target_lengths = torch.tensor([1]).repeat(len(targets_cls)).cuda()


        elif sid:
            targets_slb = torch.tensor([self.voice_idx]).repeat(slb_len).cuda()
            targets_cls = targets
            lid_w, sid_w, vad_w,asr_w = 0,1,0,0
            #print(222222, targets_slb.size(),targets_cls.size())
            targets_flat = torch.tensor([self.unk_idx]).repeat(len(targets_cls))
            target_lengths = torch.tensor([1]).repeat(len(targets_cls)).cuda()


        elif vad:
            lid_w, sid_w, vad_w,asr_w = 0,0,1,0
            targets_slb = targets
            targets_cls = targets[:targets.size(0)//150]

            targets_flat = torch.tensor([self.unk_idx]).repeat(len(targets_cls))
            target_lengths = torch.tensor([1]).repeat(len(targets_cls)).cuda()


        elif asr:
            lid_w, sid_w, vad_w, asr_w = 0, 0, 0, 1
            #slb_len = min(input_lengths) * bsz
            targets_slb = torch.tensor([self.voice_idx]).repeat(slb_len).cuda()
            targets_cls = torch.tensor([self.unk_idx]).repeat(bsz).cuda()
            targets_flat = sample["target"].masked_select(pad_mask)
            target_lengths = sample["target_lengths"]

            #print(333333,min(input_lengths),feats.size(),logits_vad.size(),targets_slb.size())
            #sys.exit()

        else:
            print('batch error',targets)
            sys.exit()

        with torch.backends.cudnn.flags(enabled=False):
            #loss_asr = self.criterion_ctc(lprobs,targets_flat,input_lengths,target_lengths)
            #loss_lid, logits_lid = self.criterion_lid(feats, targets_cls)
            loss_sid, logits_sid = self.criterion_sid(feats, targets_cls)
            loss_vad = self.criterion_vad(logits_vad.view(-1, logits_vad.size(-1)), targets_slb)
            #loss = lid_w * loss_lid + sid_w * loss_sid + vad_w * loss_vad + asr_w * loss_asr
            loss = sid_w * loss_sid + vad_w * loss_vad

        target_lengths = sample["target_lengths"]

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = 1 #sample['target'].size(0) if self.sentence_avg else ntokens #sample['ntokens']

        logging_output = {
            'loss': loss.data,
            #'loss_lid': lid_w * loss_lid.data,
            'loss_sid': sid_w * loss_sid.data,
            'loss_vad': vad_w * loss_vad.data,
            #'loss_asr': asr_w * loss_asr.data,
        }


        if lid:
            preds = logits_lid.argmax(dim=1)
            logging_output['ncorrect_lid'] = (preds == targets).sum()
            logging_output['nsentences_lid'] = sample['target'].size(0)
            logging_output['sample_size_lid'] = sample_size
            logging_output['ntokens_lid'] = ntokens
            logging_output["c_errors"] = 0
            logging_output["c_total"] = 0


        elif sid:
            preds = logits_sid.argmax(dim=1)
            logging_output['ncorrect_sid'] = (preds == targets).sum()
            logging_output['nsentences_sid'] = sample['target'].size(0)
            logging_output['sample_size_sid'] = sample_size
            logging_output['ntokens_sid'] = ntokens
            logging_output["c_errors"] = 0
            logging_output["c_total"] = 0
            
        elif vad:
            #logits = logits_vad.view(-1, logits_vad.size(-1))
            preds = logits_vad.view(-1, logits_vad.size(-1)).argmax(dim=-1)
            try:
                logging_output['ncorrect_vad'] = (preds == targets).sum()
            except:
                print('vad sample_size error')
                logging_output['ncorrect_vad'] = 0
            logging_output['nsentences_vad'] = sample['target'].size(0)
            logging_output['sample_size_vad'] = sample_size
            logging_output['ntokens_vad'] = ntokens
            logging_output["c_errors"] = 0
            logging_output["c_total"] = 0

        elif asr:
            logging_output['nsentences'] = sample['target'].size(0)
            logging_output['sample_size'] = sample_size
            logging_output['ntokens'] = ntokens

            if not model.training:
                import editdistance

                with torch.no_grad():
                    lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                    c_err = 0
                    c_len = 0
                    w_errs = 0
                    w_len = 0
                    wv_errs = 0
                    for lp, t, inp_l in zip(
                            lprobs_t,
                            sample["target_label"]
                            if "target_label" in sample
                            else sample["target"],
                            input_lengths,
                    ):
                        lp = lp[:inp_l].unsqueeze(0)

                        decoded = None
                        if self.w2l_decoder is not None:
                            decoded = self.w2l_decoder.decode(lp)
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]
                                if len(decoded) < 1:
                                    decoded = None
                                else:
                                    decoded = decoded[0]

                        p = (t != self.task.target_dictionary.pad()) & (
                                t != self.task.target_dictionary.eos()
                        )
                        targ = t[p]
                        targ_units = self.task.target_dictionary.string(targ)
                        targ_units_arr = targ.tolist()

                        toks = lp.argmax(dim=-1).unique_consecutive()
                        pred_units_arr = toks[toks != self.blank_idx].tolist()

                        c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                        c_len += len(targ_units_arr)

                        targ_words = post_process(targ_units, self.post_process).split()

                        pred_units = self.task.target_dictionary.string(pred_units_arr)
                        pred_words_raw = post_process(pred_units, self.post_process).split()

                        if decoded is not None and "words" in decoded:
                            pred_words = decoded["words"]
                            w_errs += editdistance.eval(pred_words, targ_words)
                            wv_errs += editdistance.eval(pred_words_raw, targ_words)
                        else:
                            dist = editdistance.eval(pred_words_raw, targ_words)
                            w_errs += dist
                            wv_errs += dist

                        w_len += len(targ_words)

                    logging_output["c_errors"] = c_err
                    logging_output["c_total"] = c_len
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output).view(-1).long()
        loss = F.nll_loss(lprobs, target, reduction='sum' if reduce else 'none')
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        #loss_sum_lid = sum(log.get('loss_lid', 0) for log in logging_outputs)
        loss_sum_sid = sum(log.get('loss_sid', 0) for log in logging_outputs)
        loss_sum_vad = sum(log.get('loss_vad', 0) for log in logging_outputs)
        #loss_sum_asr = sum(log.get('loss_asr', 0) for log in logging_outputs)



        #ntokens_lid = sum(log.get('ntokens_lid', 0) for log in logging_outputs)
        ntokens_sid = sum(log.get('ntokens_sid', 0) for log in logging_outputs)
        ntokens_vad = sum(log.get('ntokens_vad', 0) for log in logging_outputs)
        #ntokens_asr = sum(log.get('ntokens', 0) for log in logging_outputs)


        #nsentences_lid = sum(log.get('nsentences_lid', 0) for log in logging_outputs)
        nsentences_sid = sum(log.get('nsentences_sid', 0) for log in logging_outputs)
        nsentences_vad= sum(log.get('nsentences_vad', 0) for log in logging_outputs)
        #nsentences_asr = sum(log.get('nsentences', 0) for log in logging_outputs)

        #sample_size_lid = sum(log.get('sample_size_lid', 0) for log in logging_outputs)
        sample_size_sid = sum(log.get('sample_size_sid', 0) for log in logging_outputs)
        sample_size_vad = sum(log.get('sample_size_vad', 0) for log in logging_outputs)
        #sample_size_asr = sum(log.get('sample_size', 0) for log in logging_outputs)

        #sample_size = sample_size_lid + sample_size_sid + sample_size_vad + sample_size_asr
        sample_size =  sample_size_sid + sample_size_vad

        #prec_sum = sum(log.get('precision', 0) for log in logging_outputs)
        #metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('loss', loss_sum / sample_size , sample_size, round=3)
        #if sample_size_lid:
        #    metrics.log_scalar('loss_lid', loss_sum_lid / sample_size_lid, sample_size_lid, round=3)
        #else:
        #    metrics.log_scalar('loss_lid', loss_sum_lid / 999999999999, sample_size_lid, round=3)
            
        if sample_size_sid:
            metrics.log_scalar('loss_sid', loss_sum_sid / sample_size_sid, sample_size_sid, round=3)
        else:
            metrics.log_scalar('loss_sid', loss_sum_sid / 999999999999, sample_size_sid, round=3)

        if sample_size_vad:
            metrics.log_scalar('loss_vad', loss_sum_vad / sample_size_vad, sample_size_vad, round=3)
        else:
            metrics.log_scalar('loss_vad', loss_sum_vad / 999999999999, sample_size_vad, round=3)

        #if sample_size_asr:
        #    metrics.log_scalar('loss_asr', loss_sum_asr / sample_size_asr, sample_size_asr, round=3)
        #else:
        #    metrics.log_scalar('loss_asr', loss_sum_asr / 999999999999, sample_size_asr, round=3)


        #ncorrect_lid = sum(log.get('ncorrect_lid', 0) for log in logging_outputs)
        #if nsentences_lid:
        #    metrics.log_scalar('accuracy_lid', 100.0 * ncorrect_lid / ntokens_lid, ntokens_lid, round=1)
        #else:
        #    metrics.log_scalar('accuracy_lid', 100.0 * ncorrect_lid / 999999999999, ntokens_lid, round=1)

        ncorrect_sid = sum(log.get('ncorrect_sid', 0) for log in logging_outputs)
        if nsentences_sid:
            metrics.log_scalar('accuracy_sid', 100.0 * ncorrect_sid / ntokens_sid, ntokens_sid, round=1)
        else:
            metrics.log_scalar('accuracy_sid', 100.0 * ncorrect_sid / 999999999999, ntokens_sid, round=1)

        ncorrect_vad = sum(log.get('ncorrect_vad', 0) for log in logging_outputs)
        if nsentences_vad:
            metrics.log_scalar('accuracy_vad', 100.0 * ncorrect_vad / ntokens_vad, ntokens_vad, round=1)
        else:
            metrics.log_scalar('accuracy_vad', 100.0 * ncorrect_vad / 999999999999, ntokens_vad, round=1)

        #c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        #metrics.log_scalar("_c_errors", c_errors)
        #c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        #metrics.log_scalar("_c_total", c_total)
        #if c_total > 0:
        #    metrics.log_scalar("uer", 100.0 * c_errors/c_total,c_total,round=3)
            #metrics.log_derived(
            #    "uer",
            #    lambda meters: round(meters["_c_errors"].sum.item() * 100.0 / meters["_c_total"].sum.item(), 3)
            #    if meters["_c_total"].sum > 0
            #    else float("nan"),
            #)




    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AM_softmax(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.2, scale=30, **kwargs):
        super(AM_softmax, self).__init__()

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.W = torch.nn.Parameter(torch.randn(nOut, nClasses), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AMSoftmax m=%.3f s=%.3f' % (self.m, self.s))
    
    def extract_logits(self,x):
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        return costh


    def forward(self, x, label=None):
        #if len(x.size())==3:
        #    loss = self.ce(x.view(-1, x.size(-1)), label)
        #    return loss

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, label)
        #prec1, _ = accuracy(costh_m_s.detach().cpu(), label.detach().cpu(), topk=(1, 5))
        return loss, costh_m_s




class Uncertainty_weighed_loss(nn.Module):
    def __init__(self, criterion):
        super(Uncertainty_weighed_loss, self).__init__()
        self.log_var = torch.nn.Parameter(torch.tensor(float(0)).cuda(), requires_grad=True)
        self.criterion = criterion

    def forward(self, x, label,x_len=None,label_len=None):
        weight = torch.exp(-self.log_var)
        if x_len and label_len:
            loss = self.criterion(x,label,x_len,label_len)
        else:
            loss,costh_m_s = self.criterion(x,label,x_len)

        loss = weight * loss + self.log_var
        return loss
