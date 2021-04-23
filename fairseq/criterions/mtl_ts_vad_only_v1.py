# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import copy
import sys
import torch.nn.functional as F
import torch
import torch.nn as nn

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data.data_utils import post_process

from fairseq.logging.meters import safe_round

@register_criterion('mtl_ts_vad_only_v1')
class MTL_TS_VAD_only_v1(FairseqCriterion):

    def __init__(self, task, sentence_avg=None,class_num=None):
        super().__init__(task)


        self.sentence_avg = sentence_avg
        self.pad_idx = 2
        if task.target_dictionary:
            self.NS_idx = task.target_dictionary.symbols.index('#NS')
            self.SPK_idx = task.target_dictionary.symbols.index('#S')
        #if task.target_dictionary:
        #    self.voice_idx = task.target_dictionary.symbols.index('#S')

        #else:
        #    if not class_num:
        #        print('please set the class_num')
        #        sys.exit()


        #self.criterion_ts_vad =  nn.CrossEntropyLoss(weight=torch.FloatTensor([2,1,1])) #weight=torch.FloatTensor([2,1,1])
        self.criterion_ts_vad = WPL_loss(weights=[1,1,0.1])
        self.contrastive_loss = ContrastiveLoss(margin=0.7)
        #print(2222, 'wpl loss')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # print('sample',sample)
        vad, asr = False, False
        assert len(set(sample['task'])) == 1
        task = sample['task'][0]
        assert task == 'vad'
        vad = True
        net_output = model(**sample['net_input'])

        logits_vad = net_output['encoder_seq_out']
        features = net_output['features']
        ts_embedding = net_output['ts_embedding']

        ori_targets = model.get_targets(sample, net_output)
        ts_targets = sample["target_ts"]
        non_padding_mask = ~net_output["padding_mask"]
        input_lengths = non_padding_mask.long().sum(-1)
        #targets = ts_targets.expand(logits_vad.size(0), logits_vad.size(1))
        targets = ts_targets

        targets = targets.contiguous().view(-1).long()
        # targets = model.get_targets(sample, net_output).view(-1).long()
        lprobs = F.softmax(logits_vad, dim=-1, dtype=torch.float32)


        if vad:
            loss_ts_vad = self.criterion_ts_vad(logits_vad.view(-1, logits_vad.size(-1)), targets)
            contrasitive_loss = self.contrastive_loss(features.contiguous().view(-1, features.size(-1)), ts_embedding.contiguous().view(-1, ts_embedding.size(-1)), targets)

            loss = loss_ts_vad + contrasitive_loss
        else:
            print('batch error', targets)
            sys.exit()

        target_lengths = sample["target_lengths"]

        ntokens = len(targets)
        sample_size = 1  # sample['target'].size(0) if self.sentence_avg else ntokens #sample['ntokens']

        logging_output = {
            'loss': loss.data,
        }

        if vad:
            logging_output['loss_vad'] = loss.data
            preds = logits_vad.view(-1, logits_vad.size(-1)).argmax(dim=-1)
            #sil_cnt = (targets == 2).sum()
            ts_cnt = (targets == 0).sum()
            pred_ts_cnt = (preds == 0).sum()
            #assert sil_cnt + sp_cnt == len(targets)
            #assert sil_cnt + sp_cnt == len(preds)

            try:
                logging_output['n_pred_ts'] = pred_ts_cnt
                logging_output['n_ts'] = ts_cnt

                logging_output['ncorrect_vad'] = (preds == targets).sum()
                logging_output['ncorrect_ts'] = ((preds == 0) & (targets == 0)).sum()
                # print(111111,logging_output['ncorrect_sil']/sil_cnt,logging_output['ncorrect_sp']/sp_cnt)
                # print(1111,sil_cnt,sp_cnt,((preds == 1) & (targets == 1)).sum(),((preds == 0) & (targets == 0)).sum())
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
        total_n_ts = sum(log.get('n_ts', 0) for log in logging_outputs)
        total_n_pred_ts = sum(log.get('n_pred_ts', 0) for log in logging_outputs)

        nsentences_vad = sum(log.get('nsentences_vad', 0) for log in logging_outputs)

        sample_size_vad = sum(log.get('sample_size_vad', 0) for log in logging_outputs)
        sample_size_asr = sum(log.get('sample_size', 0) for log in logging_outputs)

        sample_size = sample_size_vad + sample_size_asr

        metrics.log_scalar('loss', loss_sum / sample_size, sample_size, round=3)

        if sample_size_vad:
            metrics.log_scalar('loss_vad', loss_sum_vad / sample_size_vad, sample_size_vad, round=3)
        else:
            metrics.log_scalar('loss_vad', loss_sum_vad / 999999999999, sample_size_vad, round=3)

        ncorrect_vad = sum(log.get('ncorrect_vad', 0) for log in logging_outputs)
        if nsentences_vad:
            metrics.log_scalar('accuracy_vad', 100.0 * ncorrect_vad / ntokens_vad, ntokens_vad, round=1)
        else:
            metrics.log_scalar('accuracy_vad', 100.0 * ncorrect_vad / 999999999999, ntokens_vad, round=1)

        if total_n_ts:
            ncorrect_ts = sum(log.get('ncorrect_ts', 0) for log in logging_outputs)
            metrics.log_scalar('recall_TS', 100.0 * ncorrect_ts / total_n_ts, total_n_ts, round=1)
        else:
            metrics.log_scalar('recall_TS', 100.0 * 0 / ntokens_vad, ntokens_vad, round=1)

        if total_n_pred_ts:
            ncorrect_ts = sum(log.get('ncorrect_ts', 0) for log in logging_outputs)
            metrics.log_scalar('precision_TS', 100.0 * ncorrect_ts / total_n_pred_ts, total_n_pred_ts, round=1)
        else:
            metrics.log_scalar('precision_TS', 100.0 * 0 / ntokens_vad, ntokens_vad, round=1)




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


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=0.7):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        label = label.clone()
        label[label>0] = 1
        cosine_similarity = F.cosine_similarity(output1, output2)
        loss_contrastive = torch.mean((1-label) * cosine_similarity +(label) * torch.clamp(self.margin - cosine_similarity, min=0.0))
        return loss_contrastive

class WPL_loss(nn.Module):
    def __init__(self,weights=[1,1,0.1]):
        super(WPL_loss, self).__init__()
        self.weights = weights
        self.nll_loss = nn.NLLLoss()
        self.w_0_1 = weights[0]
        self.w_0_2 = weights[1]
        self.w_1_2 = weights[2]

    def forward(self, z, y):
        #print(1100,z.size())
        exp_z = torch.exp(z)
        #print(111,exp_z.size())
        exp_z_T = torch.t(exp_z)
        #print(2222, exp_z_T.size())
        exp_z_T_0 = exp_z_T[0]
        exp_z_T_1 = exp_z_T[1]
        exp_z_T_2 = exp_z_T[2]
        sum_z_0_1 = exp_z_T_0 + exp_z_T_1
        sum_z_0_2 = exp_z_T_0 + exp_z_T_2
        sum_z_1_2 = exp_z_T_1 + exp_z_T_2

        WPL_0 = self.w_0_1 * torch.log(exp_z_T_0 / sum_z_0_1) + \
                self.w_0_2 * torch.log(exp_z_T_0 / sum_z_0_2)

        WPL_1 = self.w_0_1 * torch.log(exp_z_T_1 / sum_z_0_1) + \
                self.w_1_2 * torch.log(exp_z_T_1 / sum_z_1_2)

        WPL_2 = self.w_0_2 * torch.log(exp_z_T_2 / sum_z_0_2) + \
                self.w_1_2 * torch.log(exp_z_T_2 / sum_z_1_2)
        #print(33333, WPL_2.size())
        WPL_0 = WPL_0.view(1, -1)
        WPL_1 = WPL_1.view(1, -1)
        WPL_2 = WPL_2.view(1, -1)
        #print(44444,WPL_2.size())
        WPL_L = torch.cat((WPL_0, WPL_1, WPL_2), 0)
        #print(5555,WPL_L.size())
        WPL = torch.t(WPL_L)
        #print(6666,WPL.size())
        WPL_loss = self.nll_loss(WPL, y)

        return WPL_loss
