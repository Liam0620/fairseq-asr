# websocket_server.py
import math
import ctypes
import random
import traceback
import copy
import socket
import asyncio
import tornado
from tornado import gen, iostream
import tornado.web
import tornado.httpserver
import tornado.ioloop
import tornado.websocket as ws
from tornado.websocket import websocket_connect
from tornado.options import define, options
import datetime
import base64
#import logging
import sys
from multiprocessing import Process, Queue
import numpy as np
import numpy
import pickle
import json
import time, os, itertools, shutil, importlib
import torch
from fairseq import checkpoint_utils, options, utils, tasks
from fairseq.utils import import_user_module
from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder
from tornado.ioloop import IOLoop
from tornado import gen
import tornado.options
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import torch.nn as nn
import torch.nn.functional as F
from VAD_ASR.vad_asr_develop import VAD_ASR_Module
from VAD_ASR.utils import load_model,Model_Manager,save_json
from pydub.audio_segment import AudioSegment


def add_asr_eval_argument(parser):
    parser.add_argument("--kspmodel", default=None, help="sentence piece model")
    parser.add_argument(
        "--wfstlm", default=None, help="wfstlm on dictonary output units"
    )
    parser.add_argument(
        "--rnnt_decoding_type",
        default="greedy",
        help="wfstlm on dictonary\
output units",
    )
    parser.add_argument(
        "--lm-weight",
        "--lm_weight",
        type=float,
        default=0.2,
        help="weight for lm while interpolating with neural score",
    )
    parser.add_argument(
        "--rnnt_len_penalty", default=-0.5, help="rnnt length penalty on word level"
    )
    parser.add_argument(
        "--w2l-decoder", choices=["viterbi", "kenlm", "fairseqlm"], help="use a w2l decoder"
    )
    parser.add_argument("--lexicon", help="lexicon for w2l decoder")
    parser.add_argument("--unit-lm", action='store_true', help="if using a unit lm")
    parser.add_argument("--kenlm-model", "--lm-model", help="lm model for w2l decoder")
    parser.add_argument("--beam-threshold", type=float, default=25.0)
    parser.add_argument("--beam-size-token", type=float, default=100)
    parser.add_argument("--word-score", type=float, default=1.0)
    parser.add_argument("--unk-weight", type=float, default=-math.inf)
    parser.add_argument("--sil-weight", type=float, default=0.0)
    parser.add_argument(
        "--dump-emissions",
        type=str,
        default=None,
        help="if present, dumps emissions into this file and exits",
    )
    parser.add_argument(
        "--dump-features",
        type=str,
        default=None,
        help="if present, dumps features into this file and exits",
    )
    parser.add_argument(
        "--load-emissions",
        type=str,
        default=None,
        help="if present, loads emissions from this file",
    )
    parser.add_argument("--only-chinese", type=bool, default=False)
    parser.add_argument("--use-cuda", type=str, default=False)

    parser.add_argument("--min-speech", type=float, default=5,
                        help="minimum frames that can be recognized as speech",)
    parser.add_argument("--min-silence", type=float, default=15,
                        help="minimum frames that can be recognized as silience",)
    parser.add_argument("--speech-onset", type=float, default=0.55,
                        help="threshold of vad score",)

    parser.add_argument("--asr-chunk-size", type=float, default=10.0,
                        help="maximum frames before asr decoding",)
    parser.add_argument("--chunk-size", type=float, default=3.0,
                        help="vad chunk size",)
    parser.add_argument("--real-time", type=str, default=False,
                        help="if realtime decoding",)
    parser.add_argument("--realtime-step", type=float,
                        help="real time decoding frequence", )
    parser.add_argument("--step", type=float,
                        help="step", )
    parser.add_argument("--address", type=str,
                        help="port", )
    parser.add_argument("--port", type=int,
                        help="port", )
    parser.add_argument("--devices", type=str,
                        help="devices", )
    parser.add_argument("--max-users-per-device", type=int,
                        help="max-users-per-device", )

    return parser

class ExecutorBase(object):
    executor = ThreadPoolExecutor(32)

global Uid_workers
Uid_workers = dict()

class StatusHandler(ws.WebSocketHandler): #tornado.web.RequestHandler
    workers_available = 1

    def get_current_user(self):
        user = self.get_argument(name='uid', default='None')
        if user == 'None':
            user = str(random.randint(0, 100000000))
        return user

    users = dict()

    def simple_init(self):
        self.current_user = str(self.get_current_user())

    def open(self):
        self.simple_init()
        print('StatusHandler收到新的WebSocket连接',self.current_user)
        if self.current_user=='ASR_server':
            self.users[self.current_user] = self
        else:
            '''
            global Uid_workers
            if 'AI_station' in Uid_workers:
                last_workers_available = Uid_workers['AI_station']
                self.workers_available = last_workers_available
                print('super_worker_available',self.workers_available)
                Uid_workers['AI_station'] = last_workers_available

            self.users['AI_station'] = self
            '''
            self.users[self.current_user] = self

    @gen.coroutine
    def on_message(self, recv_msg):
        global Uid_workers
        if self.current_user=='ASR_server':
            self.workers_available = int(recv_msg)
            result_d = {"num_workers_available": self.workers_available, "num_requests_processed": 11}
            result_j = json.dumps(result_d)
            '''
            Uid_workers['AI_station'] = self.workers_available
            self.users['AI_station'].write_message(result_j)
            '''


            for user in self.users.keys():
                if user!='ASR_server':
                    Uid_workers[user]=self.workers_available
                    try:
                        self.users[user].write_message(result_j)
                        #print('write_to_222222222222', user,result_j)
                    except:
                        print(user,'disconnected')


        try:
            #workers_available = Uid_workers['AI_station']#Uid_workers[self.current_user]
            workers_available = Uid_workers[self.current_user]
        except Exception as e:
            workers_available =self.workers_available

        result_d = {"num_workers_available": workers_available, "num_requests_processed": 11}
        result_j = json.dumps(result_d)
        if self.current_user != 'ASR_server':
            #print('recv_11111111111',recv_msg,result_j)
            self.write_message(result_j)


    def on_close(self):
        print(time.asctime(time.localtime(time.time())), 'user:', self.current_user, 'disconnet')
        self.close()

    def check_origin(self, origin):
        return True

class web_socket_handler(ws.WebSocketHandler,ExecutorBase,Model_Manager):
    '''
    This class handles the websocket channel
    '''

    def get_current_user(self):
        user = self.get_argument(name='uid', default='None')
        if user and user != 'None':
            return user

    def get_datatype(self):
        datatype = self.get_argument(name='datatype', default='pcm')
        if datatype and not datatype is 'None':
            return datatype

    def get_sr(self):
        sr = self.get_argument(name='sr', default='16000')
        if sr and not sr is 'None':
            return sr

    def get_lang(self):
        lang = self.get_argument(name='lang', default='ma')
        if lang and not lang is 'None':
            return lang

    def get_debug_mode(self):
        debug = self.get_argument(name='debug', default='False')
        if debug and not debug is 'None':
            return debug=='True'

    def get_realtime_mode(self):
        debug = self.get_argument(name='realtime', default='True')
        if debug:
            return debug=='True'

    def get_bs_mode(self):
        repeat_bs = self.get_argument(name='repeat_bs', default='True')
        if repeat_bs:
            return repeat_bs=='True'

    def get_max_len(self):
        decode_max_len = self.get_argument(name='max_len', default=10)
        return int(decode_max_len)

    users = dict()  # 用来存放在线用户的容器

    @classmethod
    def init_all(cls,path,arg_overrides,vad_asr_args,max_users_per_device,task,use_cuda,is_large_model,devices,Url):
        cls.load_models(cls,path,arg_overrides,task,use_cuda,is_large_model,devices)
        cls.max_users = max_users_per_device*len(devices)
        cls.max_users_per_device = max_users_per_device
        cls.vad_asr_args = vad_asr_args
        cls.data_save_path = '/data3/mli2/fairseq-asr/http_TS_vad/audioAnalysis/audio'
        cls.result_path = '/data3/mli2/fairseq-asr/http_TS_vad/audioAnalysis/data'
        cls.num_available_devices = 1
        cls.Url = Url
        return cls

    def simple_init(self):
        self.Ws = None
        self.current_user = str(self.get_current_user())+'-'+time.asctime( time.localtime(time.time()) ).replace(' ','-').replace(':','_')
        self.datatype = str(self.get_datatype())
        self.sr = int(self.get_sr())
        self.lang = self.get_lang()
        self.debug = self.get_debug_mode()
        self.realtime = self.get_realtime_mode()
        self.reat_bs = self.get_bs_mode()
        self.max_decode_len = self.get_max_len()
        self.final_results = []
        self.disp_msg = ''
        self.assigned_model,self.model_id = self.get_model(self.current_user, self.max_users_per_device)
        self.max_wait = 60
        if not self.assigned_model is None:
            self.num_available_devices = self.num_available_devices - 1
            self.vad_asr_module = VAD_ASR_Module(model=self.assigned_model, **self.vad_asr_args,
                                                 real_time=self.realtime,repeat_beamsearch=self.reat_bs,max_decode_len=self.max_decode_len)

        self.audio = bytes("", 'utf-8')
        #self.result_path = os.path.join(self.data_save_path,str(self.get_current_user()))
        #os.makedirs(self.result_path,exist_ok=True)
        self.audio_path = os.path.join(self.data_save_path, '%s.wav' % (str(self.get_current_user())))
        #self.data_recv_path = os.path.join(self.result_path, '%s.txt' % (str(self.get_current_user())))
        #self.recv_f = open(self.data_recv_path, 'w')
        self.result_file = os.path.join(self.result_path,'%s.json'%(str(self.get_current_user())))
        self.result_dict = {'audio_path':self.audio_path,'result':[]}

    def open(self):
        self.last = time.time()
        self.connect_time = time.time()
        self.data_wait = 2
        self.get_msg = False
        self.simple_init()
        self.connect()
        print('收到新的WebSocket连接',self.current_user)

        self.users[self.current_user] = self  # 建立连接后添加用户到容器中
        self.loop = None
        if len(self.users)>self.max_users or self.assigned_model is None:
            print('assined_model is None',self.assigned_model is None,self.current_user)
            print('#users exceed',len(self.users)>self.max_users,self.current_user)
            msg = {'type':'error','text':"Error: the number of current online users reached the limit. Please try later"}
            msg_json = json.dumps(msg)
            self.write_message(msg_json)
            self.close()
        else:
            msg = {'type': 'status',
                   'text': "You are connected"}
            msg_json = json.dumps(msg)
            self.write_message(msg_json)
            self.loop = tornado.ioloop.PeriodicCallback(self.write_to_client, 100,
                                                        io_loop=tornado.ioloop.IOLoop.instance())
            self.loop.start()
            print('start PeriodicCallback loop')

    #'''
    @gen.coroutine
    def connect(self):
        #logging.info("trying to connect slave server")
        try:
            self.Ws = yield websocket_connect(self.Url)
            self.Ws.write_message(str(self.num_available_devices))
        except Exception as e:
            #logging.error("connection error")
            print("connection error")
    #'''



    @gen.coroutine
    def on_message(self, recv_msg):
        '''
            Message received on the handler
        '''
        self.last = time.time()
        self.get_msg = True
        try:
            if not recv_msg=='EOT':
                #self.recv_f.write(recv_msg + '\n')
                recv_msg = recv_msg.encode('utf-8')
                recv_msg = base64.b64decode(recv_msg)
                #bytes_int = int.from_bytes(recv_msg,byteorder='big',signed=False)
                #bytes_int = np.frombuffer(recv_msg, dtype=np.int32)
                #self.recv_f.write(str(bytes_int) + ' ' +str(len(bytes_int)) +'\n')
                self.audio += recv_msg
                #self.recv_f.write(str(len(recv_msg))+' ')
            else:
                print('receive EOT',self.current_user)
                #audiosegment = AudioSegment(data=self.audio, sample_width=2, frame_rate=16000,channels=1)
                #audiosegment.export(self.audio_path, format='wav')
                #self.recv_f.close()

            yield self.process_msg(recv_msg)

        except Exception as e:
            ret_error = traceback.format_exc()
            print(ret_error)
            if self.debug and not self.loop is None:
                msg = {'type': 'error',
                       'text': str(ret_error)}
                msg_json = json.dumps(msg)
                self.write_message(msg_json)
            self.on_close()


    @run_on_executor
    def process_msg(self, recv_msg):
        is_end = recv_msg=='EOT'
        # 异步执行延时操作 to do sth
        try:
            #print('EOT to vad_asr_module',self.current_user)
            self.vad_asr_module.open_asr(recv_msg,is_end)
        except AttributeError as e:
            self.close()

    @gen.coroutine
    def write_to_client(self):
        if  (time.time() - self.connect_time > self.data_wait) and not self.get_msg:
            print('Error: no data received over %d seconds after connection' % self.data_wait)
            msg = {'type': 'error',
                   'text': 'Error: no data received over %d seconds after connection' % self.data_wait}
            msg_json = json.dumps(msg)
            try:
                self.write_message(msg_json)
            except tornado.websocket.WebSocketClosedError as e:
                print('write_end_WebSocketClosedError')
                self.loop.stop()
            self.close()

        if time.time() - self.last > self.max_wait:
            print('Error: no data received over %d seconds'%self.max_wait)
            msg = {'type': 'error',
                   'text': 'Error: no data received over %d seconds'%self.max_wait}
            msg_json = json.dumps(msg)
            try:
                self.write_message(msg_json)
            except tornado.websocket.WebSocketClosedError as e:
                print('write_end_WebSocketClosedError')
                self.loop.stop()
            self.close()

        msg = yield self.write2client()
        if not msg is None:
            print('return_result:',msg,self.current_user)
            self.last = time.time()
            if msg['type']=='final_result':
                print('sentence:' + msg['text'] + ' from:' + self.current_user + '\n')
                self.result_dict['result'].append(msg)

            ret_msg = json.dumps(msg,ensure_ascii=False)
            try:
                self.write_message(ret_msg)
            except tornado.websocket.WebSocketClosedError as e:
                self.close()


    @run_on_executor
    def write2client(self):
        # 异步执行延时操作 to do sth
        msg = self.vad_asr_module.get_result()
        return msg

    def on_close(self):
        if self.current_user in self.users.keys():
            if self.model_id is not None:
                self.num_available_devices = self.num_available_devices + 1
                while self.Ws is None:
                    time.sleep(0.1)

                self.Ws.write_message(str(self.num_available_devices))

            print(time.asctime(time.localtime(time.time())), 'user:', self.get_current_user(), 'disconnet')
            self.release_model(self.model_id, self.current_user)
            if self.current_user in self.users.keys():
                del self.users[self.current_user]  # 用户关闭连接后从容器中移除用户

            if self.model_id is not None:
                if (not self.current_user in self.users.keys()) and (self.current_user in self.models[self.model_id]['users']):
                    self.models[self.model_id]['users'].remove(self.current_user)
                    assert self.current_user not in self.models[self.model_id]['users']

            if len(self.users)==0:
                self.release_all()


            #if self.model_id is not None:
            #    assert self.current_user not in self.models[self.model_id]['users']
            audiosegment = AudioSegment(data=self.audio, sample_width=2, frame_rate=16000,channels=1)
            audiosegment.export(self.audio_path, format='wav')
            final_result = []
            for i, item in enumerate(self.result_dict['result']):
                if i == 0:
                    if item['start'] > 0:
                        final_result.append({'text': '<NTS>', 'start': 0, 'end': item['start']})
                    del item['type']
                    final_result.append(item)
                else:
                    prev_seg_end = self.result_dict['result'][i - 1]['end']
                    start = item['start']
                    if start < prev_seg_end:
                        final_result[-1]['end'] = start
                        item['start'] = prev_seg_end
                        prev_seg_end = start

                    if item['start'] > prev_seg_end:
                        final_result.append({'text': '<NTS>', 'start': prev_seg_end, 'end': item['start']})
                    del item['type']
                    final_result.append(item)
            self.result_dict['result'] = final_result
            # print(self.result_dict)
            save_json(self.result_file, self.result_dict)

            print('current users online:', len(self.users), self.users.keys())
            for k,v in self.models.items():
                print(self.models[k]['device'],self.models[k]['users'])

            if not self.loop is None:
                self.loop.stop()
        self.close()

        if not self.Ws is None:
            self.Ws.close()

    def check_origin(self, origin):
        return True



def main(args):
    #logging.basicConfig(
    #    level=logging.DEBUG,
    #    format='%(name)s: %(message)s',
    #    stream=sys.stderr,
    #)

    import_user_module(args)
    use_cuda = args.use_cuda=='True'

    task = tasks.setup_task(args)
    generator = W2lKenLMDecoder(args, task.target_dictionary)
    vad_asr_args =  {
        'min_speech_len':args.min_speech,
        'min_sil_len':args.min_silence,
        'asr_chunk_seconds': args.asr_chunk_size,
        'segment_duration': args.chunk_size,
        'step':args.step,
        'onset':args.speech_onset,
        'right_buffer':0.64,
        'left_buffer':0.64,
        'use_cuda':use_cuda,
        'generator':generator,
        'detail': True,
    }
    devices = args.devices.split(',')
    devices = list(map(int, devices))
    heart_beats_ws = 'ws://%s:%s/server/speech/status?uid=ASR_server' % (args.address, args.port) #

    server_args = {
        'path': args.path,
        'arg_overrides': args.model_overrides,
        'vad_asr_args':vad_asr_args,
        'max_users_per_device':args.max_users_per_device,
        'task': task,
        'use_cuda':use_cuda,
        'is_large_model':True,
        'devices':devices,
        'Url':heart_beats_ws
    }


    app = tornado.web.Application(
        handlers=[
            (r'/server/speech/realtime',web_socket_handler.init_all(**server_args)),
            (r'/server/speech/status', StatusHandler),  # .init_all()
            ]
        )
    # setup the server
    server = tornado.httpserver.HTTPServer(app) 
    server.listen(args.port,address=args.address)
    print('start listening')
    # start io/event loop
    tornado.ioloop.IOLoop.instance().start()



def make_parser():
    parser = options.get_generation_parser()
    parser = add_asr_eval_argument(parser)
    return parser

def cli_main():
    parser = make_parser()
    args = options.parse_args_and_arch(parser)
    main(args)

if __name__ == "__main__":
    cli_main()
