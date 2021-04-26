import numpy as np
import math
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
import queue
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
    def __init__(self,model,min_speech_len,min_sil_len,asr_chunk_seconds, segment_duration,step,
                 onset,right_buffer,left_buffer,use_cuda,generator,real_time,detail,repeat_beamsearch,max_decode_len):
        self.min_speech_len = min_speech_len/0.02
        self.min_sil_len = min_sil_len/0.02
        self.asr_chunk_seconds = asr_chunk_seconds
        self.segment_duration = segment_duration
        self.step = step
        self.onset = onset
        self.use_cuda = use_cuda
        self.generator = generator
        self.model = model['model']
        self.device = model['device']
        self.use_cuda = use_cuda
        self.continuous_preds = []
        self.tmp_continuous_preds = []
        self.tmp_result = ''
        self.realtime_tmp = ''
        self.asr_chunk_list = []
        self.mem_asr_chunk = None
        self.prev_status = 0
        self.sample_rate = 16000
        self.asr_chunk_size = int(asr_chunk_seconds * self.sample_rate)
        self.vad_chunk_size = int(segment_duration * self.sample_rate)
        self.step_size = int(step * self.sample_rate)

        self.left_buffer = min(int(self.sample_rate * left_buffer),self.step_size)
        self.right_buffer = min(int(self.sample_rate * right_buffer),self.step_size)
        self.vad_chunk_bytes = bytes("", 'utf-8')
        self.is_real_time = real_time
        self.cnn_frames = 320
        self.result_queue = Queue(maxsize=0)
        self.recv_queue = Queue(maxsize=0)
        self.thread_list = []
        self.batch_cnt = 0
        self.is_wait = 0
        self.max_vad_sz = self.step_size * 2
        self.data_split_sz = 2560 * 2
        self.min_process_len = 320 * 2
        self.is_processing = False
        self.detail = detail
        self.repeat_beamsearch = repeat_beamsearch
        self.current_frame = 0
        self.is_speech = 0
        self.max_decode_len = max_decode_len/0.02
        self.max_seg_len = 30/0.02

        # speed test


        self.first_batch = 1
        self.init_time = time.time()

        embedding_path = '/data3/mli2/fairseq-asr/http_TS_vad/speakers/limeng/limeng.npy'
        sp_emb = np.load(embedding_path)
        self.ts_embedding = torch.from_numpy(sp_emb)
        print(1111111,embedding_path)


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

                #if is_break:
                #    break
        else:
            recv_info = {'data':recv_msg,'is_end':is_end}
            self.recv_queue.put(recv_info)
            if not self.is_processing:
                self.process_T = MyThread(self.process, {'arg': None}, 'Process_Thread')
                self.process_T.daemon = True
                self.process_T.start()


    def process(self,arg=None):
        self.is_processing = True
        while 1:
            try:
                recv_info = self.recv_queue.get()
            except: #queue.Empty as e:
                time.sleep(0.1)
                recv_info = self.recv_queue.get()
            recv_msg = recv_info['data']
            is_end = recv_info['is_end']
            ret_msg = self.online_decoding(recv_msg, is_end)
            if ret_msg['text']:
                self.result_queue.put(ret_msg)
            if is_end:
                break
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
        else:
            print('process final chunk')
            send_msg,utt_length = self.process_final_chunk()# + '<->'
            send_msg += '<->'
            # get timeline of segment _by_mli
            seg_start = (self.current_frame - utt_length) / self.sample_rate
            seg_end = self.current_frame / self.sample_rate
            segment = (seg_start, seg_end)
            eos = True

        if len(self.vad_chunk_bytes) >= self.vad_chunk_size * 2:
            audio_chunk = Byt2Arr(self.vad_chunk_bytes)
            # sliding step by_mli
            try:
                self.last_hist_vad_chunk_len = self.hist_vad_chunk_len
            except:
                pass
            self.vad_chunk_bytes = self.vad_chunk_bytes[self.step_size*2:]
            self.hist_vad_chunk_len = int(len(self.vad_chunk_bytes)/2)

            batch = torch.from_numpy(audio_chunk).float().view(1, -1)
            if self.use_cuda:
                device = torch.device("cuda:%s" % (self.device))
                batch = batch.to(device)
                ts_embedding = self.ts_embedding.to(device)

            speech_logits = self.vad(batch, ts_embedding)
            if self.first_batch:
                self.first_batch = 0
                current_step_logits = speech_logits[:,:int(self.step_size / self.cnn_frames),:]
                self.prev_logits = copy.deepcopy(speech_logits)

            else:
                current_step_logits = speech_logits[:,:int(self.step_size / self.cnn_frames),:] + self.prev_logits[:,int(self.step_size / self.cnn_frames):int(self.step_size*2 / self.cnn_frames),:]
                part_prev_logits = self.prev_logits[:,int(self.step_size / self.cnn_frames):,:]
                pad_zeros = torch.zeros(
                                        [current_step_logits.size(0), speech_logits.size(1)-part_prev_logits.size(1), current_step_logits.size(2)]
                                        , dtype=part_prev_logits.dtype, device=part_prev_logits.device)
                part_prev_logits = torch.cat(( part_prev_logits,pad_zeros), 1)
                self.prev_logits = copy.deepcopy(speech_logits)+part_prev_logits

                batch = batch[:, :int(self.step_size)]

            speech_seq = self.vad_score(current_step_logits)


            if speech_seq.sum() < self.min_speech_len:
                # silence chunk
                send_msg,eos,utt_length = self.process_sil_chunk(batch)
                # get timeline of segment _by_mli
                if eos:
                    seg_end = self.current_frame-self.hist_vad_chunk_len - self.min_sil_len*self.cnn_frames
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
                        seg_end = self.current_frame+ end_frame-self.hist_vad_chunk_len- self.min_sil_len*self.cnn_frames
                        seg_start = seg_end - utt_length - indent
                        segment = (seg_start/ self.sample_rate, seg_end/ self.sample_rate)

                    if eos:
                        type = 'final_result'
                    else:
                        type = 'temp_result'
                    ret_msg = {'type': type, 'text': send_msg}
                    # get timeline of segment _by_mli
                    if eos:
                        ret_msg['start'] = segment[0]
                        ret_msg['end'] = segment[1]
                    return ret_msg



        if eos:
            type = 'final_result'
        else:
            type = 'temp_result'
        ret_msg = {'type': type, 'text': send_msg}
        # get timeline of segment _by_mli
        if eos:
            ret_msg['start'] = segment[0]
            ret_msg['end'] = segment[1]
        return ret_msg

    def vad(self,batch,input_embeddings):
        encoder_inp1 = {'source': batch, 'padding_mask': None,'input_embeddings':input_embeddings}
        net_output = self.model(**encoder_inp1, stage='cnn_only')
        logits = net_output['encoder_seq_out']
        return logits
        #lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        #data = lprobs.view(-1, lprobs.size(2))
        #data = torch.exp(data[:, 0])
        #speech_seq = (data > self.onset).long().cpu().numpy()
        #return speech_seq
    def vad_score(self,logits):
        lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
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
        #print(11111, batch.size())
        if self.prev_status:
            if len(self.asr_chunk_list) > 0:
                self.asr_chunk_list.append(batch[:, :self.right_buffer])
                asr_chunk = torch.cat(self.asr_chunk_list, 1)
                #print(22222,asr_chunk.size())
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
            self.realtime_tmp = ''
            self.mem_asr_chunk = None

        self.prev_status = 0
        return send_msg,eou,utt_length

    def process_spk_chunk(self,batch):
        send_msg = ''
        self.asr_chunk_list.append(batch)
        eos = False
        asr_chunk = torch.cat(self.asr_chunk_list, 1)
        self.prev_status = 1
        #'''
        if len(self.asr_chunk_list) == 1 and self.prev_status == 0 and self.is_real_time:
            tmp_asr_out = self.asr(asr_chunk, mid_result=True)
            tmp_hyps = self.tmp_decode(tmp_asr_out, self.generator)
            if tmp_hyps:
                send_msg = tmp_hyps
        #'''

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

                self.realtime_tmp = hyps
                #print(22222, self.realtime_tmp)

                if hyps:
                    send_msg = hyps

        else:
            if self.is_real_time and len(self.asr_chunk_list)>1:
                tmp_asr_out = self.asr(asr_chunk, mid_result=True)
                tmp_hyps = self.tmp_decode(tmp_asr_out, self.generator)
                if tmp_hyps:
                    send_msg = self.realtime_tmp + tmp_hyps
                    #print(33333333,self.realtime_tmp,tmp_hyps)
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
                    self.realtime_tmp = ''
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
                        self.realtime_tmp = ''
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
        #try:
        if 1:
            asr_out = self.asr(asr_chunk)
            self.continuous_preds.append(asr_out)
            hyps,_ = self.decode(self.continuous_preds, self.generator)
        #except RuntimeError as e:
        #    hyps = ''
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

