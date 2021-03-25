import numpy as np
import numpy
import copy
import torch
from fairseq import checkpoint_utils, options, utils, tasks
from fairseq.utils import import_user_module
from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder
import torch.nn as nn
import torch.nn.functional as F
from VAD_ASR.utils import load_model,post_process,Byt2Arr
from multiprocessing import Process, Queue
import time
import logging
import threading
from threading import Thread

class MyThread(Thread):
    def __init__(self,func,kwargs,name):
        Thread.__init__(self)
        self.name = name
        self.func = func
        self.kwargs = kwargs

    def run(self):
        self.func(**self.kwargs)

class VAD_ASR_Module():
    def __init__(self,model,min_speech_len,min_sil_len,asr_chunk_seconds, segment_duration,
                 onset,right_buffer,left_buffer,use_cuda,generator,real_time,detail,repeat_beamsearch,max_decode_len):
        self.min_speech_len = min_speech_len/0.02
        self.min_sil_len = min_sil_len/0.02
        self.asr_chunk_seconds = asr_chunk_seconds
        self.segment_duration = segment_duration
        self.onset = onset
        self.use_cuda = use_cuda
        self.generator = generator
        self.model = model['model']
        self.device = model['device']
        self.use_cuda = use_cuda
        self.continuous_preds = []
        self.tmp_continuous_preds = []
        self.tmp_result = ''
        self.asr_chunk_list = []
        self.mem_asr_chunk = None
        self.prev_status = 0
        self.sample_rate = 16000
        self.asr_chunk_size = asr_chunk_seconds * self.sample_rate
        self.vad_chunk_size = int(segment_duration * self.sample_rate)
        self.left_buffer = int(self.sample_rate * left_buffer)
        self.right_buffer = int(self.sample_rate * right_buffer)
        self.vad_chunk_bytes = bytes("", 'utf-8')
        self.is_real_time = real_time
        self.cnn_frames = 320
        self.result_queue = Queue(maxsize=0)
        self.recv_queue = Queue(maxsize=0)
        self.thread_list = []
        self.current_process_num = 0
        self.is_wait = 0
        self.max_vad_sz = 10240 * 2
        self.data_split_sz = 2560 * 2
        self.min_process_len = 320 * 2
        self.is_processing = False
        self.detail = detail
        self.repeat_beamsearch = repeat_beamsearch
        self.current_frame = 0
        self.is_speech = 0
        self.max_decode_len = max_decode_len/0.02
        self.prev_batch = None
        self.max_seg_len = 30/0.02

    def open_asr(self,recv_msg,is_end):
        is_break = False
        if len(recv_msg) > self.max_vad_sz:
            for i in range(0, len(recv_msg), self.data_split_sz):
                recv_msg_part = recv_msg[i:i + self.data_split_sz]
                next_recv_msg_part = recv_msg[i + self.data_split_sz:i + 2 * self.data_split_sz]
                if len(next_recv_msg_part) <= self.min_process_len:
                    recv_msg_part = recv_msg[i:]
                    is_break = True
                recv_info = {'data': recv_msg_part, 'is_end': is_end}
                self.recv_queue.put(recv_info)
                if not self.is_processing:
                    self.process_T = MyThread(self.process, {'arg': None}, 'Process_Thread')
                    self.process_T.start()
                if is_break:
                    break
        else:
            recv_info = {'data':recv_msg,'is_end':is_end}
            self.recv_queue.put(recv_info)
            if not self.is_processing:
                self.process_T = MyThread(self.process, {'arg': None}, 'Process_Thread')
                self.process_T.start()

    def process(self,arg=None):
        while not self.recv_queue.empty():
            self.is_processing = True
            recv_info = self.recv_queue.get(block=False)
            recv_msg = recv_info['data']
            is_end = recv_info['is_end']
            ret_msg = self.online_decoding(recv_msg, is_end)
            if ret_msg['text']:
                self.result_queue.put(ret_msg)
        self.is_processing = False
        return

    def get_result(self):
        if not self.result_queue.empty():
            msg = self.result_queue.get()
        else:
            msg = None
        return msg

    @torch.no_grad()
    def online_decoding(self,recv_msg,is_end=False):
        send_msg = ''
        eos = False
        if not is_end:
            self.current_frame += len(recv_msg) / 2
            self.vad_chunk_bytes += recv_msg

        if len(self.vad_chunk_bytes) >= self.vad_chunk_size * 2:
            audio_chunk = Byt2Arr(self.vad_chunk_bytes)
            self.vad_chunk_bytes = bytes("", 'utf-8')
            batch = torch.from_numpy(audio_chunk).float().view(1, -1)
            if self.use_cuda:
                device = torch.device("cuda:%s" % (self.device))
                batch = batch.to(device)
            if not self.prev_batch is None:
                batch = torch.cat( (self.prev_batch,batch),1 )

            speech_seq = self.vad(batch)

            if speech_seq.sum() < self.min_speech_len:
                # silence chunk
                send_msg,eos,utt_length = self.process_sil_chunk(batch)
                # get timeline of segment _by_mli
                if eos:
                    seg_end = self.current_frame - self.vad_chunk_size - self.min_sil_len*self.cnn_frames #/ self.sample_rate
                    seg_start = seg_end - utt_length
                    segment = (max(0, seg_start)/ self.sample_rate, seg_end/ self.sample_rate)

            elif len(speech_seq) - speech_seq.sum() < self.min_speech_len:
                # speaking chunk
                send_msg,eos = self.process_spk_chunk(batch)
            else:
                # sil&spk mix chunk
                for infos in self.process_chunk(batch,speech_seq):
                    send_msg, eos, utt_length, end_frame,indent = infos
                    # get timeline of segment _by_mli
                    if eos:
                        seg_end = self.current_frame- self.vad_chunk_size + end_frame - self.min_sil_len*self.cnn_frames
                        seg_start = seg_end - utt_length - indent
                        segment = (seg_start/ self.sample_rate, seg_end/ self.sample_rate)

                    if eos:
                        type = 'final_result'
                    else:
                        type = 'temp_result'
                    ret_msg = {'type': type, 'text': send_msg}
                    # get timeline of segment _by_mli
                    if eos:
                        ret_msg['segment']=segment
                    return ret_msg

        if is_end:
            send_msg,utt_length = self.process_final_chunk()# + '<->'
            send_msg += '<->'
            # get timeline of segment _by_mli
            seg_start = (self.current_frame - utt_length) / self.sample_rate
            seg_end = self.current_frame / self.sample_rate
            segment = (seg_start, seg_end)

            eos = True
        if eos:
            type = 'final_result'
        else:
            type = 'temp_result'
        ret_msg = {'type': type, 'text': send_msg}
        # get timeline of segment _by_mli
        if eos:
            ret_msg['segment'] = segment
        return ret_msg

    def vad(self,batch):
        encoder_inp1 = {'source': batch, 'padding_mask': None}
        net_output = self.model(**encoder_inp1, stage='cnn_only')
        seq_label = net_output['encoder_seq_out']
        lprobs = F.log_softmax(seq_label, dim=-1, dtype=torch.float32)
        data = lprobs.view(-1, lprobs.size(2))
        data = torch.exp(data[:, 0])
        speech_seq = (data > self.onset).long().cpu().numpy()
        return speech_seq

    def asr(self,asr_chunk,mid_result=False):
        encoder_inp = {'source': asr_chunk, 'padding_mask': None}
        if mid_result:
            tmp_asr_out = self.model(**encoder_inp)['encoder_out'].cpu()
            return tmp_asr_out
        asr_out = self.model(**encoder_inp)['encoder_out'].cpu().numpy()
        return asr_out

    def process_sil_chunk(self,batch):
        send_msg = ''
        eou = False
        utt_length = None
        self.prev_batch = None
        if self.prev_status:
            if len(self.asr_chunk_list) > 0:
                self.asr_chunk_list.append(batch[:, :self.right_buffer])
                asr_chunk = torch.cat(self.asr_chunk_list, 1)
                asr_out = self.asr(asr_chunk)
                self.continuous_preds.append(asr_out)
            hyps,_ = self.decode(self.continuous_preds, self.generator)
            self.asr_chunk_list = []
            if hyps:
                send_msg = hyps
                eou = True
                utt_length = len(numpy.vstack(self.continuous_preds)) * self.cnn_frames

            self.continuous_preds = []
            self.tmp_continuous_preds = []
            self.tmp_result = ''
            self.mem_asr_chunk = None

        self.prev_status = 0
        return send_msg,eou,utt_length

    def process_spk_chunk(self,batch):
        send_msg = ''
        self.asr_chunk_list.append(batch)
        self.prev_batch = None
        eos = False
        asr_chunk = torch.cat(self.asr_chunk_list, 1)
        self.prev_status = 1
        if len(self.asr_chunk_list) == 1 and self.prev_status == 0 and self.is_real_time:
            tmp_asr_out = self.asr(asr_chunk, mid_result=True)
            tmp_hyps = self.tmp_decode(tmp_asr_out, self.generator)
            if tmp_hyps:
                send_msg = tmp_hyps

        if asr_chunk.size(1) >= self.asr_chunk_size + self.right_buffer:

            real_chunk_sz = asr_chunk.size(1) - batch.size(1)

            if self.mem_asr_chunk is None:
                mem_flag = 0
                self.mem_asr_chunk = copy.deepcopy(asr_chunk[:, :real_chunk_sz + self.right_buffer])
            else:
                self.mem_asr_chunk = torch.cat((self.mem_asr_chunk[:,
                                                -self.left_buffer - self.right_buffer:-self.right_buffer],
                                                asr_chunk[:, :real_chunk_sz + self.right_buffer]), 1)
                mem_flag = 1

            asr_out = self.asr(self.mem_asr_chunk)
            if mem_flag:
                asr_out = asr_out[-int(self.right_buffer / self.cnn_frames) - int(real_chunk_sz / self.cnn_frames):-int(
                    self.right_buffer / self.cnn_frames), :, :]
            else:
                asr_out = asr_out[:int(real_chunk_sz / self.cnn_frames), :, :]

            self.continuous_preds.append(asr_out)
            self.asr_chunk_list = [batch]

            continuous_preds = numpy.vstack(self.continuous_preds)
            len_conti_preds = len(continuous_preds)

            if len_conti_preds>self.max_seg_len:
                continuous_preds = torch.FloatTensor(continuous_preds)
                hyps = self.tmp_decode(continuous_preds,self.generator)
                self.continuous_preds = []
                self.tmp_continuous_preds = []
                eos=True
                if hyps:
                    send_msg = hyps
                return send_msg,eos

            if self.is_real_time:
                if self.repeat_beamsearch:
                    self.tmp_continuous_preds.append(asr_out)
                    hyps,_len_preds = self.decode(self.tmp_continuous_preds, self.generator)
                    if _len_preds>self.max_decode_len:
                        self.tmp_continuous_preds = []
                        self.tmp_result += hyps + ' '
                        hyps = self.tmp_result
                    else:
                        hyps = self.tmp_result + hyps + ' '

                else:
                    hyps,_ = self.decode([asr_out], self.generator)
                    self.tmp_result += hyps+' '
                    hyps = self.tmp_result

                if hyps:
                    send_msg = hyps

        return send_msg,eos

    def process_chunk(self,batch,speech_seq):
        send_msg = ''
        eou = False
        utt_length = None
        continuous_spk = []
        continuous_nospk = []
        cnt_nospk = 0
        end_frame = 0
        indent = 0
        res = []
        for j, label in enumerate(speech_seq):
            if label == 1:
                continuous_spk.append(j)
            else:
                continuous_nospk.append(label)
                continuous_spk.append(j)
            # silence
            if len(continuous_nospk) - cnt_nospk > self.min_sil_len:
                if len(continuous_spk) > (self.min_sil_len + self.min_speech_len):
                    self.asr_chunk_list.append(batch[:,
                                               max(0, int(
                                                   continuous_spk[
                                                       0] * self.cnn_frames - self.sample_rate * 0.01)):int(
                                                   continuous_spk[-1] * self.cnn_frames)])
                    asr_chunk = torch.cat(self.asr_chunk_list, 1)
                    asr_out = self.asr(asr_chunk)
                    self.continuous_preds.append(asr_out)
                    continuous_spk = []
                    continuous_nospk = []
                    cnt_nospk = 0
                    self.asr_chunk_list = []
                    hyps,_ = self.decode(self.continuous_preds, self.generator)
                    if hyps:
                        send_msg = hyps
                        eou = True
                        utt_length = len(numpy.vstack(self.continuous_preds)) * self.cnn_frames
                        end_frame = j * self.cnn_frames
                    self.continuous_preds = []
                    self.tmp_continuous_preds = []
                    self.tmp_result = ''
                    self.mem_asr_chunk = None

                else:
                    continuous_spk = []
                    cnt_nospk = len(continuous_nospk)
                    if len(self.asr_chunk_list) > 0:
                        asr_chunk = torch.cat(self.asr_chunk_list, 1)
                        asr_out = self.asr(asr_chunk)
                        self.continuous_preds.append(asr_out)
                        hyps,_ = self.decode(self.continuous_preds, self.generator)
                        self.asr_chunk_list = []
                        if hyps:
                            send_msg = hyps
                            eou = True
                            utt_length = len(numpy.vstack(self.continuous_preds)) * self.cnn_frames
                            end_frame = j * self.cnn_frames
                            indent = end_frame

                        self.continuous_preds = []
                        self.tmp_continuous_preds = []
                        self.tmp_result = ''
                        self.mem_asr_chunk = None

                    self.prev_status = 0
                res.append((send_msg, eou, utt_length, end_frame,indent))

        if len(continuous_spk) - (len(continuous_nospk) - cnt_nospk) > self.min_speech_len:
            self.asr_chunk_list.append(
                batch[:, max(0, int(continuous_spk[0] * self.cnn_frames - self.sample_rate * 0.01)):int(
                    continuous_spk[-1] * self.cnn_frames)])
            if len(self.asr_chunk_list) == 1 and self.prev_status == 0 and self.is_real_time:
                asr_chunk = torch.cat(self.asr_chunk_list, 1)
                tmp_asr_out = self.asr(asr_chunk, mid_result=True)
                tmp_hyps = self.tmp_decode(tmp_asr_out, self.generator)
                if tmp_hyps:
                    send_msg = tmp_hyps

            self.prev_status = 1
        else:
            self.prev_status = 0
        res.append((send_msg, eou, utt_length, end_frame,indent))
        return res

    def process_final_chunk(self):
        audio_chunk = Byt2Arr(self.vad_chunk_bytes)
        batch = torch.from_numpy(audio_chunk).float().view(1, -1)
        if self.use_cuda:
            device = torch.device("cuda:%s" % (self.device))
            batch = batch.to(device)
        self.asr_chunk_list.append(batch)
        asr_chunk = torch.cat(self.asr_chunk_list, 1)
        try:
            asr_out = self.asr(asr_chunk)
            self.continuous_preds.append(asr_out)
            hyps,_ = self.decode(self.continuous_preds, self.generator)
        except RuntimeError as e:
            hyps = ''
        try:
            utt_length = len(numpy.vstack(self.continuous_preds)) * self.cnn_frames
        except ValueError as e:
            utt_length = 0
        send_msg = hyps
        return send_msg,utt_length

    def decode(self,continuous_preds, generator):
        continuous_preds = numpy.vstack(continuous_preds)
        len_preds = len(continuous_preds)
        continuous_preds = torch.FloatTensor(continuous_preds)
        hypo = generator.decode(continuous_preds.transpose(0, 1).contiguous())[0][0]
        hyp_words = " ".join(hypo["words"]).replace('[LAUGHTER]', '').replace('[NOISE]', '').replace(
            '[VOCALIZED-NOISE]',
            '').strip()
        res = hyp_words.replace(' ', '')
        hyp_words = ' '.join(res).strip().replace('  ', ' ').replace('< u n k >', '<unk>').strip()
        return hyp_words,len_preds

    def tmp_decode(self,continuous_preds, generator):
        hypo = generator.decode(continuous_preds.transpose(0, 1).contiguous())[0][0]
        hyp_words = " ".join(hypo["words"]).replace('[LAUGHTER]', '').replace('[NOISE]', '').replace(
            '[VOCALIZED-NOISE]',
            '').strip()
        res = hyp_words.replace(' ', '')
        hyp_words = ' '.join(res).strip().replace('  ', ' ').replace('< u n k >', '<unk>').strip()
        return hyp_words

