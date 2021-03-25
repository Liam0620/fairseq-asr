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

@register_criterion('mtl_vad_only')
class MTL_VAD_only(FairseqCriterion):

    def __init__(self, task, sentence_avg=None,class_num=None):
        super().__init__(task)


        self.sentence_avg = sentence_avg

        if task.target_dictionary:
            self.voice_idx = task.target_dictionary.symbols.index('#S')

        else:
            if not class_num:
                print('please set the class_num')
                sys.exit()


        self.criterion_vad =  nn.CrossEntropyLoss()


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        #print('sample',sample)
        vad,asr = False,False
        assert len(set(sample['task'])) == 1
        task = sample['task'][0]
        assert task=='vad'
        vad = True


        net_output = model(**sample['net_input'])
        logits_vad = net_output['encoder_seq_out']

        targets = model.get_targets(sample, net_output).view(-1).long()

        if vad:
            targets = targets-self.voice_idx
            loss = self.criterion_vad(logits_vad.view(-1, logits_vad.size(-1)), targets)

        else:
            print('batch error',targets)
            sys.exit()


        target_lengths = sample["target_lengths"]

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = 1 #sample['target'].size(0) if self.sentence_avg else ntokens #sample['ntokens']

        logging_output = {
            'loss': loss.data,
        }

        if vad:
            logging_output['loss_vad'] = loss.data
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
        loss_sum_vad = sum(log.get('loss_vad', 0) for log in logging_outputs)

        ntokens_vad = sum(log.get('ntokens_vad', 0) for log in logging_outputs)


        nsentences_vad= sum(log.get('nsentences_vad', 0) for log in logging_outputs)


        sample_size_vad = sum(log.get('sample_size_vad', 0) for log in logging_outputs)
        sample_size_asr = sum(log.get('sample_size', 0) for log in logging_outputs)

        sample_size = sample_size_vad + sample_size_asr

        metrics.log_scalar('loss', loss_sum / sample_size , sample_size, round=3)

        if sample_size_vad:
            metrics.log_scalar('loss_vad', loss_sum_vad / sample_size_vad, sample_size_vad, round=3)
        else:
            metrics.log_scalar('loss_vad', loss_sum_vad / 999999999999, sample_size_vad, round=3)


        ncorrect_vad = sum(log.get('ncorrect_vad', 0) for log in logging_outputs)
        if nsentences_vad:
            metrics.log_scalar('accuracy_vad', 100.0 * ncorrect_vad / ntokens_vad, ntokens_vad, round=1)
        else:
            metrics.log_scalar('accuracy_vad', 100.0 * ncorrect_vad / 999999999999, ntokens_vad, round=1)





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



class Uncertainty_weighed_loss(nn.Module):
    def __init__(self, criterion):
        super(Uncertainty_weighed_loss, self).__init__()
        self.log_var = torch.nn.Parameter(torch.zeros(1).cuda(), requires_grad=True)
        self.weight = torch.exp(-self.log_var)
        self.criterion = criterion
    def forward(self, x, label):
        loss = self.criterion(x,label)
        loss = loss * self.weight + self.log_var
        return loss

class Uncertainty_weighed_loss_ctc(nn.Module):
    def __init__(self, criterion):
        super(Uncertainty_weighed_loss_ctc, self).__init__()
        self.log_var = torch.nn.Parameter(torch.zeros(1).cuda(), requires_grad=True)
        self.criterion = criterion
        self.weight = torch.exp(-self.log_var)

    def forward(self, x, label,x_len,label_len):
        loss = self.criterion(x,label,x_len,label_len)
        loss = loss * self.weight + self.log_var
        return loss
