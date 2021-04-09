__author__ = 'tanel'

import argparse
from ws4py.client.threadedclient import WebSocketClient
import time
import threading
import sys
import urllib
import queue as Queue
import json
import time
import os
import wave
import base64
import random
import datetime
import librosa
import soundfile as sf

class MyClient(WebSocketClient):

    def __init__(self, audiofile, url, protocols=None, extensions=None, heartbeat_freq=None, byterate=32000,
                 save_adaptation_state_filename=None, send_adaptation_state_filename=None):
        super(MyClient, self).__init__(url, protocols, extensions, heartbeat_freq)
        self.final_hyps = ''
        self.audiofile = audiofile
        self.byterate = byterate
        self.final_hyp_queue = Queue.Queue()
        self.save_adaptation_state_filename = save_adaptation_state_filename
        self.send_adaptation_state_filename = send_adaptation_state_filename
        self.time = time.time()
    def send_data(self, data):
        self.send(data)

    def opened(self):
        #print "Socket opened!"
        def send_data_to_ws():
            wf = wave.open(self.audiofile, 'rb') # /data3/mli2/asr_vad_test.wav
            params = wf.getparams()
            nchannels, sampwidth, framerate, nframes = params[:4]
            data = wf.readframes(nframes)# * 10
            CHUNK = 3200
            #print(len(data))
            if 1:
                for i in range(0, len(data), CHUNK):
                    audio_chunk = data[i:i + CHUNK]
                    audio_chunk = base64.b64encode(audio_chunk)  # base64编码
                    #print('1111111111:' + str(i))
                    messageDict = {}
                    messageDict['type'] = "wav_data"
                    messageDict['data'] = audio_chunk.decode('utf-8')
                    #print(audio_chunk)
                    #time.sleep(0.1)
                    self.send(json.dumps(messageDict))
                    #self.send(audio_chunk)
            print("Audio sent, now sending EOT")
            messageDict = {}
            messageDict['type'] = "wav_data"
            messageDict['data'] = "EOT"
            self.send(json.dumps(messageDict))
            #self.send("EOT")

        t = threading.Thread(target=send_data_to_ws)
        t.start()


    def received_message(self, m):

        print("22222:" +  str(m),time.time()-self.time)
        self.time = time.time()
        try:
            jsonObj = json.loads(str(m))
            self.final_hyps = str(jsonObj['text'])
            if str(m).find("<->") != -1:
                self.close()
        except:
            print("error:" + str(m))
        #print >> sys.stderr, "RESPONSE:", response
        #print >> sys.stderr, "JSON was:", m



    def get_full_hyp(self, timeout=60):
        return self.final_hyp_queue.get(timeout)

    def closed(self, code, reason=None):
        print("Websocket closed() called")
        #print >> sys.stderr
        self.final_hyp_queue.put(self.final_hyps)


def main():
    starttime = datetime.datetime.now()
    cur_file = '/data3/mli2/asr_vad/miss_example.wav'
    #cur_file = '/data3/mli2/test_asr/test/ht_7499_16k/ht216_android_speaker3472_0254.wav'
    # ws = MyClient(cur_file, 'ws://172.18.30.90:6008/server/speech/realtime?uid=%s&realtime=False'%f, byterate=16000)
    #for i in range(2):
    if 1:
        ws = MyClient(cur_file,
                      'ws://172.18.30.66:8083/aiserver/speech/realtime?&realtime=True&appkey=123&uid=%s&type=asr&lang=ch&model=i' % (
                                  'mli' + str(random.randint(0, 1000000))), byterate=16000)
        ws.connect()
        result = ws.get_full_hyp()

    endtime = datetime.datetime.now()
    print("time:" + str((endtime - starttime).seconds))





    sys.exit()
    set = 'cn_great_16k'
    rootdir = "/data3/mli2/test_asr/test/%s"%set #"/data/syzhou/work/data/tmp/test/ztspeech_16k"  #"/data3/mli2/test_asr/test/ztspeech_16k"
    source_sr = rootdir.split('_')[-1]
    text_path = os.path.join(rootdir, 'text')
    text_dict = {}
    ref_path = 'test_hyps_refs/%s.ref' % set
    hyp_path = 'test_hyps_refs/%s.hyp' % set
    ref_f = open(ref_path,'w')
    hyp_f = open(hyp_path,'w')

    with open(text_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            id = line.split(' ')[0]
            text = line.split(' ')[1:]
            text = ' '.join(list(''.join(text).replace(' ', '').replace('[NOISE]', '').strip()))
            text_ref = text + ' ' + id + '\n'
            text_dict[id]=text_ref

    for root, dirs, files in os.walk(rootdir):
        for f in files:
            cur_file = os.path.join(root,f)
            if cur_file[-4:]=='.wav':
                if source_sr == '8k':
                    print('8k_to_16k',cur_file)
                    src_sig, sr = sf.read(cur_file)  # name是要 输入的wav 返回 src_sig:音频数据  sr:原采样频率
                    if sr != 16000:
                        dst_sig = librosa.resample(src_sig, sr, 16000)  # resample 入参三个 音频数据 原采样频率 和目标采样频率
                        sf.write(cur_file, dst_sig, 16000)  #
                id = f.replace('.wav','')
                if id in text_dict:
                    print(cur_file,id)
                    cur_file = '/data3/mli2/asr_vad/miss_example.wav'
                    #ws = MyClient(cur_file, 'ws://172.18.30.90:6008/server/speech/realtime?uid=%s&realtime=False'%f, byterate=16000)
                    ws = MyClient(cur_file,
                                  'ws://172.18.30.66:8083/aiserver/speech/realtime?&realtime=False&appkey=123&uid=%s&type=asr&lang=ch&model=i' % (f+str(random.randint(0, 1000000))), byterate=16000)
                    ws.connect()
                    result = ws.get_full_hyp()
                    hyp = result.replace('<->','')
                    if hyp and ('Error' not in hyp):
                        hyp = hyp + ' '+ id + '\n'
                        ref = text_dict[id]
                        hyp_f.write(hyp)
                        ref_f.write(ref)
                        print('hyp:',hyp,'ref:',ref)

    ref_f.close()
    hyp_f.close()
    print('caculating wer')
    sys_command = "wer --tail-ids %s %s" % (ref_path, hyp_path)
    os.system(sys_command)
    endtime = datetime.datetime.now()
    print("time:" + str((endtime - starttime).seconds))

if __name__ == "__main__":
    main()

