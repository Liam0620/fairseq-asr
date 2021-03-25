# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import logging
import numpy as np
import random
import itertools

import torch
import torch.nn.functional as F

from .. import FairseqDataset

logger = logging.getLogger(__name__)
normalize_wav = lambda wav: wav / (np.sqrt(np.mean(wav ** 2)) + 1e-8)

class RawAudioDataset(FairseqDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.min_length = min_length
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats, curr_sample_rate):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[start:end]

    def collater(self, samples):
        samples = [
            s
            for s in samples
            if s["source"] is not None
        ]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask
        return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input, "task":[s["task"] for s in samples]}
        #"input_embedding": [s["input_embedding"] for s in samples]

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

class RawAudioDataset_mtl(FairseqDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.min_length = min_length
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats, curr_sample_rate):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[start:end]

    def collater(self, samples):
        samples = [
            s
            for s in samples
            if s["source"] is not None
        ]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask
        return {"id": torch.LongTensor([s["id"] for s in samples]),"rand_id": torch.LongTensor([s["rand_id"] for s in samples]), "net_input": input}

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

class FileAudioDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
        noise_path=None,
        snr_min=5,
        snr_max=20,
        noise_prob=0.5,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            min_length=min_length,
            pad=pad,
            normalize=normalize,
        )

        self.fnames = []
        self.segments = []
        self.noise_path = noise_path #'/data3/mli2/mli/noise_dataset'
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.noise_prob = noise_prob
        skipped = 0
        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            if 'MTL' in manifest_path:
                self.task = manifest_path.split('/')[-2].split('_')[0]
            else:
                self.task = None

            for line in f:
                items = line.strip().split("\t")
                assert len(items) == 2 or len(items) == 4, line
                if len(items) == 2:
                    sz = int(items[1])
                elif len(items) == 4:
                    sz = int(items[3])
                    segment_span = (int(items[1]),int(items[2]))
                    self.segments.append(segment_span)

                if min_length is not None and sz < min_length:
                    skipped += 1
                    continue
                self.fnames.append(items[0])
                self.sizes.append(sz)
        self.noise_files=None
        if self.noise_path and self.task in ['scd','vad']:
            self.noise_files=[]
            for dirpath, dirnames, filenames in os.walk(self.noise_path):
                for name in filenames:
                    self.noise_files.append(os.path.join(dirpath, name))

        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")


    def __getitem__(self, index):
        import soundfile as sf
        fname = os.path.join(self.root_dir, self.fnames[index])
        wav, curr_sample_rate = sf.read(fname)
        #print(111,fname)
        if self.segments:
            segment_span = self.segments[index]
            start = segment_span[0]
            end = segment_span[1]
            wav = wav[start:end]
        wav_len = len(wav)
        if not self.noise_files is None and random.random()<=self.noise_prob:
            noise_file = random.choice(self.noise_files)
            noise,noise_sr = sf.read(noise_file)
            if len(noise.shape)>1:
                noise = noise[:,0]
                #print(noise.shape)
            len_noise = len(noise)
            assert noise_sr==curr_sample_rate
            while len_noise<wav_len:
                noise =np.concatenate((noise,noise),axis=0)
                len_noise = len(noise)
            rand_noise_start = random.randint(0, len_noise - wav_len)
            crop_noise = noise[rand_noise_start:rand_noise_start + wav_len]
            noise = normalize_wav(crop_noise)
            snr = (self.snr_max - self.snr_min) * np.random.random_sample() + self.snr_min
            alpha = np.exp(-np.log(10) * snr / 20)
            wav = wav + alpha * noise



        feats = torch.from_numpy(wav).float()
        #feats = torch.where(feats==0, torch.full_like(feats, 1e-5), feats)

        feats = self.postprocess(feats, curr_sample_rate)
        #feats = torch.where(feats == 0, torch.full_like(feats, 1e-8), feats)

        if torch.any(torch.isnan(feats)):
            print('is_nan',fname)
            sys.exit()
        if torch.any(torch.isinf(feats)):
            print('isinf',fname)
            sys.exit()

        return {"id": index, "source": feats,'task':self.task}

class FileAudioDataset_ts(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        ts_emb_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
        noise_path=None,
        snr_min=5,
        snr_max=20,
        noise_prob=0.5,
        ts_prob=0.5,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            min_length=min_length,
            pad=pad,
            normalize=normalize,
        )

        self.fnames = []
        self.segments = []
        self.noise_path = noise_path #'/data3/mli2/mli/noise_dataset'
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.noise_prob = noise_prob
        self.ts_prob = ts_prob
        self.speaker_embeddings = {}
        speakers = os.listdir(ts_emb_path)
        for speaker in speakers:
            emb_path = os.path.join(ts_emb_path,speaker,'%s.npy'%speaker)
            sp_emb = np.load(emb_path)
            sp_emb = torch.from_numpy(sp_emb)
            self.speaker_embeddings[speaker] = sp_emb

        skipped = 0
        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            if 'MTL' in manifest_path:
                self.task = manifest_path.split('/')[-2].split('_')[0]
            else:
                self.task = None

            for line in f:
                items = line.strip().split("\t")
                assert len(items) == 2 or len(items) == 4, line
                if len(items) == 2:
                    sz = int(items[1])
                elif len(items) == 4:
                    sz = int(items[3])
                    segment_span = (int(items[1]),int(items[2]))
                    self.segments.append(segment_span)

                if min_length is not None and sz < min_length:
                    skipped += 1
                    continue
                self.fnames.append(items[0])
                self.sizes.append(sz)
        self.noise_files=None
        if self.noise_path and self.task in ['scd','vad']:
            self.noise_files=[]
            for dirpath, dirnames, filenames in os.walk(self.noise_path):
                for name in filenames:
                    self.noise_files.append(os.path.join(dirpath, name))

        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")


    def __getitem__(self, index):
        import soundfile as sf

        fname = os.path.join(self.root_dir, self.fnames[index])
        wav, curr_sample_rate = sf.read(fname)
        #print(111,fname)
        if self.segments:
            segment_span = self.segments[index]
            start = segment_span[0]
            end = segment_span[1]
            wav = wav[start:end]
        wav_len = len(wav)
        if not self.noise_files is None and random.random()<=self.noise_prob:
            noise_file = random.choice(self.noise_files)
            noise,noise_sr = sf.read(noise_file)
            if len(noise.shape)>1:
                noise = noise[:,0]
                #print(noise.shape)
            len_noise = len(noise)
            assert noise_sr==curr_sample_rate
            while len_noise<wav_len:
                noise =np.concatenate((noise,noise),axis=0)
                len_noise = len(noise)
            rand_noise_start = random.randint(0, len_noise - wav_len)
            crop_noise = noise[rand_noise_start:rand_noise_start + wav_len]
            noise = normalize_wav(crop_noise)
            snr = (self.snr_max - self.snr_min) * np.random.random_sample() + self.snr_min
            alpha = np.exp(-np.log(10) * snr / 20)
            wav = wav + alpha * noise



        feats = torch.from_numpy(wav).float()
        #feats = torch.where(feats==0, torch.full_like(feats, 1e-5), feats)

        feats = self.postprocess(feats, curr_sample_rate)
        #feats = torch.where(feats == 0, torch.full_like(feats, 1e-8), feats)

        if torch.any(torch.isnan(feats)):
            print('is_nan',fname)
            sys.exit()
        if torch.any(torch.isinf(feats)):
            print('isinf',fname)
            sys.exit()
        ts_label = None
        input_embedding = None
        if self.task=='vad':
            cur_spk = self.fnames[index].split('/')[0]
            if random.random()<=self.ts_prob and cur_spk in self.speaker_embeddings.keys():
                input_embedding=self.speaker_embeddings[cur_spk]
                ts_label=torch.tensor([0],dtype=torch.int32)

            elif cur_spk not in self.speaker_embeddings.keys():
                random_spk = random.choice(list(self.speaker_embeddings.keys()))
                input_embedding = self.speaker_embeddings[random_spk]
                ts_label = torch.tensor([2], dtype=torch.int32)
            else:
                random_spk = random.choice(list(self.speaker_embeddings.keys()))
                input_embedding = self.speaker_embeddings[random_spk]
                if random_spk==cur_spk:
                    ts_label=torch.tensor([0],dtype=torch.int32)
                else:
                    ts_label=torch.tensor([1],dtype=torch.int32)
                #print(2222,input_embedding.size(),random_spk,cur_spk)

                #sys.exit()
        return {"id": index, "source": feats,'task':self.task,'input_embedding':input_embedding,'ts_label':ts_label}
