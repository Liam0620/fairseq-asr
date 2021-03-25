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
from fairseq.criterions import FairseqCriterion, register_criterion,LegacyFairseqCriterion


@register_criterion('mtl_classification_v0')
class MTL_ClassifyCriterion_v0(FairseqCriterion):

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        num_classes = len(task.target_dictionary.symbols)
        self.criterion_sid = AM_softmax(768, num_classes, margin=0.2, scale=30)
        self.criterion_lid = AM_softmax(768, num_classes, margin=0.2, scale=30)
        #self.criterion_vad = nn.NLLLoss(reduction='sum')
        self.criterion_vad =  nn.CrossEntropyLoss() #

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        #print('sample',sample)
        net_output = model(**sample['net_input'])

        feats = net_output["features_out"]
        logits_vad = net_output['encoder_seq_out']

        targets = model.get_targets(sample, net_output).view(-1).long()
        lid = targets.le(13).all() and targets.gt(2).all()

        sid = targets.gt(13).all() and targets.le(1224).all()

        vad = targets.gt(1224).all() and targets.le(1226).all()

        slb_len = logits_vad.view(-1, logits_vad.size(-1)).size(0)
        #print(9999999999999,slb_len)

        if lid:
            targets_slb = torch.tensor([1225]).repeat(slb_len).cuda()
            targets_cls = targets
            lid_w, sid_w, vad_w = 1,0,0
            #print(111111,targets_slb.size(),targets_cls.size())

        elif sid:
            targets_slb = torch.tensor([1225]).repeat(slb_len).cuda()
            targets_cls = targets
            lid_w, sid_w, vad_w = 0,1,0
            #print(222222, targets_slb.size(),targets_cls.size())

        elif vad:
            lid_w, sid_w, vad_w = 0,0,1
            targets_slb = targets
            targets_cls = targets[:targets.size(0)//150]
            #print(333333, targets_slb.size(),targets_cls.size())
        else:
            print('batch error',targets)
            sys.exit()

        loss_lid, logits_lid = self.criterion_lid(feats, targets_cls)
        loss_sid, logits_sid = self.criterion_sid(feats, targets_cls)


        try:
            loss_vad = self.criterion_vad(logits_vad.view(-1, logits_vad.size(-1)), targets_slb)
        except:
            print('vad logits&targets size mismatch',logits_vad.size(),logits_vad.view(-1, logits_vad.size(-1)).size(),targets_slb.size())
            loss_vad = 0 * loss_lid

        loss = lid_w*loss_lid+ sid_w*loss_sid+ vad_w*loss_vad


        '''
        if lid:
            loss, logits = self.criterion_lid(feats,targets)

        elif sid:
            loss, logits = self.criterion_sid(feats, targets)

        elif vad:
            logits = net_output['encoder_seq_out']
            #lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            #lprobs = lprobs.view(-1, lprobs.size(-1))
            loss = self.criterion_vad(logits.view(-1, logits.size(-1)), targets)
        '''


        target_lengths = sample["target_lengths"]
        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample['target'].size(0) if self.sentence_avg else ntokens #sample['ntokens']

        #preds_lid = logits_lid.argmax(dim=1)
        #ncorrect_lid = lid_w * (preds_lid == targets).sum()

        #preds_sid = logits_sid.argmax(dim=1)
        #ncorrect_sid = sid_w * (preds_sid == targets).sum()

        #preds_vad = logits_vad.view(-1, logits_vad.size(-1)).argmax(dim=-1)
        #ncorrect_vad = vad_w * (preds_vad == targets).sum()

        logging_output = {
            'loss': loss.data,
            'loss_lid': lid_w * loss_lid.data,
            'loss_sid': sid_w * loss_sid.data,
            'loss_vad': vad_w * loss_vad.data,
            #'ncorrect_lid':ncorrect_lid,
            #'ncorrect_sid':ncorrect_sid,
            #'ncorrect_vad':ncorrect_vad,
        }


        if lid:
            preds = logits_lid.argmax(dim=1)
            logging_output['ncorrect_lid'] = (preds == targets).sum()
            logging_output['nsentences_lid'] = sample['target'].size(0)
            logging_output['sample_size_lid'] = sample_size
            logging_output['ntokens_lid'] = ntokens
            '''
            logging_output['ncorrect_sid'] = 0
            logging_output['nsentences_sid'] = 0
            logging_output['sample_size_sid'] = 0
            logging_output['ntokens_sid'] = 0

            logging_output['ncorrect_vad'] = 0
            logging_output['nsentences_vad'] = 0
            logging_output['sample_size_vad'] = 0
            logging_output['ntokens_vad'] = 0
            '''


        elif sid:
            preds = logits_sid.argmax(dim=1)
            logging_output['ncorrect_sid'] = (preds == targets).sum()
            logging_output['nsentences_sid'] = sample['target'].size(0)
            logging_output['sample_size_sid'] = sample_size
            logging_output['ntokens_sid'] = ntokens
            '''
            logging_output['ncorrect_lid'] = 0
            logging_output['nsentences_lid'] = 0
            logging_output['sample_size_lid'] = 0
            logging_output['ntokens_lid'] = 0

            logging_output['ncorrect_vad'] = 0
            logging_output['nsentences_vad'] = 0
            logging_output['sample_size_vad'] = 0
            logging_output['ntokens_vad'] = 0
            '''
            
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

            '''
            logging_output['ncorrect_lid'] = 0
            logging_output['nsentences_lid'] = 0
            logging_output['sample_size_lid'] = 0
            logging_output['ntokens_lid'] = 0

            logging_output['ncorrect_sid'] = 0
            logging_output['nsentences_sid'] = 0
            logging_output['sample_size_sid'] = 0
            logging_output['ntokens_sid'] = 0
            '''

        #logging_output['precision'] = prec1.data
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        #lprobs2 = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1).long()
        #print(11111,target.size(),222,lprobs.size(),sample_size)
        loss = F.nll_loss(lprobs, target, reduction='sum' if reduce else 'none')
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        loss_sum_lid = sum(log.get('loss_lid', 0) for log in logging_outputs)
        loss_sum_sid = sum(log.get('loss_sid', 0) for log in logging_outputs)
        loss_sum_vad = sum(log.get('loss_vad', 0) for log in logging_outputs)


        ntokens_lid = sum(log.get('ntokens_lid', 0) for log in logging_outputs)
        ntokens_sid = sum(log.get('ntokens_sid', 0) for log in logging_outputs)
        ntokens_vad = sum(log.get('ntokens_vad', 0) for log in logging_outputs)

        nsentences_lid = sum(log.get('nsentences_lid', 0) for log in logging_outputs)
        nsentences_sid = sum(log.get('nsentences_sid', 0) for log in logging_outputs)
        nsentences_vad= sum(log.get('nsentences_vad', 0) for log in logging_outputs)

        sample_size_lid = sum(log.get('sample_size_lid', 0) for log in logging_outputs)
        sample_size_sid = sum(log.get('sample_size_sid', 0) for log in logging_outputs)
        sample_size_vad = sum(log.get('sample_size_vad', 0) for log in logging_outputs)

        sample_size = sample_size_lid + sample_size_sid + sample_size_vad



        #prec_sum = sum(log.get('precision', 0) for log in logging_outputs)
        #metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('loss', loss_sum / sample_size , sample_size, round=3)
        if sample_size_lid:
            metrics.log_scalar('loss_lid', loss_sum_lid / sample_size_lid, sample_size_lid, round=3)
        else:
            metrics.log_scalar('loss_lid', loss_sum_lid / 999999999999, sample_size_lid, round=3)
            
        if sample_size_sid:
            metrics.log_scalar('loss_sid', loss_sum_sid / sample_size_sid, sample_size_sid, round=3)
        else:
            metrics.log_scalar('loss_sid', loss_sum_sid / 999999999999, sample_size_sid, round=3)

        if sample_size_vad:
            metrics.log_scalar('loss_vad', loss_sum_vad / sample_size_vad, sample_size_vad, round=3)
        else:
            metrics.log_scalar('loss_vad', loss_sum_vad / 999999999999, sample_size_vad, round=3)


        ncorrect_lid = sum(log.get('ncorrect_lid', 0) for log in logging_outputs)
        if nsentences_lid:
            metrics.log_scalar('accuracy_lid', 100.0 * ncorrect_lid / ntokens_lid, ntokens_lid, round=1)
        else:
            metrics.log_scalar('accuracy_lid', 100.0 * ncorrect_lid / 999999999999, ntokens_lid, round=1)

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


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

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





'''
class MTL_ClassifyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

        num_classes = len(task.target_dictionary.symbols)
        self.criterion_sid = AM_softmax(768, num_classes, margin=0.2, scale=30)
        self.criterion_lid = AM_softmax(768, num_classes, margin=0.2, scale=30)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample['net_input'])
        feats = net_output["features_out"]
        targets = model.get_targets(sample, net_output).view(-1).long()

        #logits = net_output["encoder_out"]
        #lprobs = model.get_normalized_probs(net_output, log_probs=True)
        #loss2 = F.nll_loss(lprobs, targets, reduction='sum' if reduce else 'none')
        #print('feats',feats,feats.size(),'targets',targets,targets.size())
        #print(111111,targets)

        mask_sid = targets.gt(13)
        num_select = mask_sid.sum()
        sid_targets = torch.masked_select(targets, mask_sid)
        feats_mask_sid = mask_sid.view(mask_sid.size(0),1).repeat(1,feats.size(1))
        sid_feats = torch.masked_select(feats, feats_mask_sid).view(num_select,feats.size(1))

        mask_lid = targets.le(13)
        #print(222222,mask_lid)
        num_select = mask_lid.sum()
        lid_targets = torch.masked_select(targets, mask_lid)
        feats_mask_lid = mask_lid.view(mask_lid.size(0),1).repeat(1,feats.size(1))
        lid_feats = torch.masked_select(feats, feats_mask_lid).view(num_select,feats.size(1))

        #sys.exit()

        if sid_targets.size(0) > 0:
            loss_sid, logits_sid = self.criterion_sid(sid_feats,sid_targets)
        else:
            loss_sid, logits_sid = 0,None

        if lid_targets.size(0) > 0:
            loss_lid, logits_lid = self.criterion_lid(lid_feats, lid_targets)
        else:
            loss_lid, logits_lid = 0,None


        loss = loss_sid + loss_lid


        target_lengths = sample["target_lengths"]
        target_lengths_sid = torch.masked_select(target_lengths, mask_sid)
        target_lengths_lid = torch.masked_select(target_lengths, mask_lid)

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        ntokens_sid = (
            target_lengths_sid.sum().item()
        )
        ntokens_lid = (
            target_lengths_lid.sum().item()
        )



        sample_size = sample['target'].size(0) if self.sentence_avg else ntokens #sample['ntokens']
        sample_size_sid = ntokens_sid
        sample_size_lid = ntokens_lid


        logging_output = {
            'loss': loss.data,
            'loss_sid': loss_sid if type(loss_sid) == int else loss_sid.data,
            'loss_lid': loss_lid if type(loss_lid) == int else loss_lid.data,
            'ntokens': ntokens,
            'ntokens_sid': ntokens_sid,
            'ntokens_lid': ntokens_lid,
            'sample_size': sample_size,
            'sample_size_sid': sample_size_sid,
            'sample_size_lid': sample_size_lid,
            'nsentences': sample_size,
        }
        #preds = logits.argmax(dim=1)
        #logging_output['ncorrect'] = (preds == targets).sum()
        if logits_sid is not None:
            preds_sid = logits_sid.argmax(dim=1)
            logging_output['ncorrect_sid'] = (preds_sid == sid_targets).sum()
        #else:
            #logging_output['ncorrect_sid'] = 0

        if logits_lid is not None:
            preds_lid = logits_lid.argmax(dim=1)
            logging_output['ncorrect_lid'] = (preds_lid == lid_targets).sum()
        #else:
            #logging_output['ncorrect_lid'] = 0
        #logging_output['precision'] = prec1.data
        #print("logging_output",logging_output)
        #print('loss',loss,logging_output)
        print(1111111,loss,logging_output)

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        #lprobs2 = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1).long()
        #print(11111,target.size(),222,lprobs.size(),sample_size)
        loss = F.nll_loss(lprobs, target, reduction='sum' if reduce else 'none')
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        loss_sum_sid = sum(log.get('loss_sid', 0) for log in logging_outputs)
        loss_sum_lid = sum(log.get('loss_lid', 0) for log in logging_outputs)

        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        sample_size_sid = sum(log.get('sample_size_sid', 0) for log in logging_outputs)
        sample_size_lid = sum(log.get('sample_size_lid', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size , sample_size, round=3)
        if sample_size_sid==0:
            metrics.log_scalar('loss_sid', 0, sample_size_sid ,round=1)
        else:
            metrics.log_scalar('loss_sid', loss_sum_sid / sample_size_sid, sample_size_sid, round=3)
        if sample_size_lid==0:
            metrics.log_scalar('loss_lid', 0, sample_size_lid, round=1)
        else:
            metrics.log_scalar('loss_lid', loss_sum_lid / sample_size_lid, sample_size_lid, round=3)

        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect_sid' in logging_outputs[0] and 'ncorrect_lid' in logging_outputs[0]:
            #ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            #metrics.log_scalar('accuracy', 100.0 * ncorrect / sample_size, sample_size, round=1)

            ncorrect_sid = sum(log.get('ncorrect_sid', 0) for log in logging_outputs)
            if sample_size_sid==0:
                metrics.log_scalar('accuracy_sid', 0.00, sample_size_sid, round=1)
            else:
                metrics.log_scalar('accuracy_sid', 100.0 * ncorrect_sid / sample_size_sid, sample_size_sid, round=1)

            ncorrect_lid = sum(log.get('ncorrect_lid', 0) for log in logging_outputs)

            if sample_size_lid==0:
                metrics.log_scalar('accuracy_lid', 0.00, sample_size_lid, round=1)
            else:
                metrics.log_scalar('accuracy_lid', 100.0 * ncorrect_lid / sample_size_lid, sample_size_lid, round=1)


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

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
'''
