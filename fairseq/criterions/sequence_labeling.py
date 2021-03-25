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
import numpy as np

@register_criterion('sequence_labeling')
class SeqLabeling_Criterion(FairseqCriterion):

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        #self.criterion = AM_softmax(768, 6116, margin=0.2, scale=30)
        self.criterion = nn.NLLLoss(reduction='sum')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        #print('sample',sample)
        net_output = model(**sample['net_input'])
        logits = net_output['encoder_seq_out']
        #logits = torch.where(torch.isnan(logits), torch.full_like(logits, 1e-8), logits)
        targets = model.get_targets(sample, [logits]).view(-1).long()

        #loss,logits,targets = self.self_cross_entropy(logits,targets)

        lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss = self.criterion(lprobs, targets)

        target_lengths = sample["target_lengths"]
        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )
        #ntokens = targets.size(0)

        sample_size = targets.size(0) if self.sentence_avg else ntokens #sample['ntokens']

        logging_output = {
            'loss': loss.data,
            'ntokens': ntokens,
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        logits = logits.view(-1,logits.size(-1))


        preds = logits.argmax(dim=-1)
        logging_output['ncorrect'] = (preds == targets).sum()
        #logging_output['precision'] = prec1.data
        #print(111111,logging_output)
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_seq_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1).long()
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss, target

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        #prec_sum = sum(log.get('precision', 0) for log in logging_outputs)
        #metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('loss', loss_sum / ntokens , ntokens, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / ntokens, ntokens, round=1)
        #metrics.log_scalar('prec_average', prec_sum / sample_size, nsentences, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def self_cross_entropy(self,input, target, ignore_index=None):
        '''自己用pytorch实现cross_entropy，
           有时候会因为各种原因，如：样本问题等，出现个别样本的loss为nan，影响模型的训练，
           不适用于所有样本loss都为nan的情况
           input:n*categ
           target:n
        '''
        ori_input = input
        input = input.contiguous().view(-1, input.size(-1))
        log_prb = F.log_softmax(input, dim=1)

        one_hot = torch.zeros_like(input).scatter(1, target.view(-1, 1), 1)  # 将target转换成one-hot编码
        loss = -(one_hot * log_prb).sum(dim=1)  # n,得到每个样本的loss

        # if ignore_index:  # 忽略[PAD]的label
        #    non_pad_mask = target.ne(0)
        #    loss = loss.masked_select(non_pad_mask)

        not_nan_mask = ~torch.isnan(loss)  # 找到loss为非nan的样本
        not_nan_mask_logits = ~torch.isnan(input)

        loss = loss.masked_select(not_nan_mask).sum()

        new_logits = input.masked_select(not_nan_mask_logits)
        new_logits = new_logits.view(-1,ori_input.size(-1))

        new_targets = target.masked_select(not_nan_mask)

        return loss,new_logits,new_targets

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

    def forward(self, x, label=None):
        #print(11111,x.size(),2222,label.size())
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
