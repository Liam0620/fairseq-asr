import os
from fairseq import checkpoint_utils, options, utils, tasks
import torch
import torch.nn.functional as F
import numpy as np
import sys
import json

class Model_Manager(object):
    models = {}
    def load_models(self,path,arg_overrides,task,use_cuda,is_large_model,devices):
        for i,device_id in enumerate(devices):
            model_name = 'model_%d'%(i)
            model = load_model(path, arg_overrides=eval(arg_overrides), task=task, is_large_model=is_large_model)
            if use_cuda:
                device = torch.device("cuda:%s"%(device_id))
                model.to(device)
            model.eval()
            self.models[model_name] = {'model':model,'device':device_id,'users':set()}

    def get_model(self,current_user,max_users_per_device):
        for model_id, model in self.models.items():
            if len(model['users']) < max_users_per_device:
                self.models[model_id]['users'].add(current_user)
                model_id = model_id
                assigned_model = model
                return assigned_model, model_id
        return None,None

    def release_model(self,model_id,current_user):
        if not model_id is None:
            if current_user in self.models[model_id]['users']:
                self.models[model_id]['users'].remove(current_user)

    def release_all(self):
        for model_id,infos in self.models.items():
            self.models[model_id]['users']=set()

def load_model(filename, arg_overrides=None, task=None,is_large_model=False):
    if not os.path.exists(filename):
        raise IOError("Model file not found: {}".format(filename))
    state = checkpoint_utils.load_checkpoint_to_cpu(filename, arg_overrides)
    args = state["args"]
    is_KD = False
    if 'KD' in filename:
        is_KD = True
    #print(1111,filename,is_KD)
    #sys.exit()
    if is_large_model and not is_KD:
        args.final_dim = 768
        args.encoder_layers= 24
        args.encoder_embed_dim= 1024
        args.encoder_ffn_embed_dim= 4096
        args.encoder_attention_heads= 16
        args.w2v_path2 = '/data3/mli2/mli/fairseq-master/examples/wav2vec/pretrained_models/Big_model/checkpoint_last.pt'
    if is_KD:
        args.encoder_attention_heads = 16
        args.w2v_path2 = 'kd_model/checkpoint_last.pt.nokd'
        args.w2v_path = 'kd_model/w2v_base/checkpoint_last.pt'

    if task is None:
        task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=True)
    return model

def load_models(filename, arg_overrides=None, task=None):
    if not os.path.exists(filename):
        raise IOError("Model file not found: {}".format(filename))
    state = checkpoint_utils.load_checkpoint_to_cpu(filename, arg_overrides)
    args = state["args"]
    if task is None:
        task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=True)
    return model

def post_process(wav,normalize=False):
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1,-1)
    return feats

def Byt2Arr(vad_chunk_bytes):
    audio_data = np.frombuffer(vad_chunk_bytes, dtype=np.int16)
    audio_chunk = audio_data / 32768
    return audio_chunk


def save_json(path, item):
    # 先将字典对象转化为可写入文本的字符串
    item = json.dumps(item,ensure_ascii=False,sort_keys=True,indent=4)
    try:
        with open(path, "w", encoding='utf-8') as f:
            f.write(item + "\n")
            print("write success")
    except Exception as e:
        print("write error==>", e)
