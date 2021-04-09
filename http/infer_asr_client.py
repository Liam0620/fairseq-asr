# websocket_client.py
import base64
import sys
import time
import wave

from tornado.ioloop import IOLoop, PeriodicCallback
from tornado import gen
from tornado.websocket import websocket_connect

global st_time
st_time = time.time()
class Client(object):
    def __init__(self, url, timeout):
        self.url = url
        self.timeout = timeout
        self.ioloop = IOLoop.instance()
        self.ws = None
        self.connect()
        PeriodicCallback(self.keep_alive, 500).start()
        self.ioloop.start()
        self.st_time = time.time()
    @gen.coroutine
    def connect(self):
        print("trying to connect")
        try:
            self.ws = yield websocket_connect(self.url)
        except Exception as e:
            print("connection error")
        else:
            print("connected")
            self.send_msg()
            self.run()

    @gen.coroutine
    def run(self):
        st = time.time()
        global st_time
        while True:
            msg = yield self.ws.read_message()
            if msg is None:
                print("connection closed")
                self.ws = None
                break
            else:
                print(msg,time.time()-st_time)

    @gen.coroutine
    def send_msg(self):
        wf = wave.open(r'/data3/mli2/asr_vad/miss_example.wav',
                       'rb')  # /data3/mli2/asr_vad_test.wav
        params = wf.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        data = wf.readframes(nframes)  # * 10
        CHUNK = 1280  # * 2 3200
        # '''
        if 1:
            for i in range(0, len(data), CHUNK):
                audio_chunk = data[i:i + CHUNK]  # chunk大小的字节流
                audio_chunk = base64.b64encode(audio_chunk)  # base64编码的bytes
                audio_chunk = audio_chunk.decode('utf-8')  # utf-8编码的string
                #time.sleep(2)
                self.ws.write_message(audio_chunk)

        self.ws.write_message('EOT')  # 结束消息

    def keep_alive(self):
        if self.ws is None:
            self.connect()
            # self.ws.write_message("keep alive")


if __name__ == "__main__":
    client = Client(
        "ws://172.18.30.90:5005/server/speech/realtime?uid=%s&realtime=False&repeat_bs=True&max_len=10" % sys.argv[1],
        5)  # &realtime=False #&datatype=wav&sr=16000&lang=ma
