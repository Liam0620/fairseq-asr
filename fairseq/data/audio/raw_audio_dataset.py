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
import librosa
import soundfile as sf
import torch
import torch.nn.functional as F

from pyannote.core import Segment,SlidingWindow,SlidingWindowFeature

from .. import FairseqDataset

logger = logging.getLogger(__name__)
normalize_wav = lambda wav: wav / (np.sqrt(np.mean(wav ** 2)) + 1e-8)
sw = SlidingWindow(duration=0.025, step=0.020)

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
        return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input,
                "task": [s["task"] for s in samples]}

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
            noise_prob=0.85,
    ):
        # print(1111,noise_prob)
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            min_length=min_length,
            pad=pad,
            normalize=normalize,
        )
        self.max_sample_size = max_sample_size
        self.min_sample_size = min_sample_size
        self.fnames = []
        self.segments = []
        self.noise_path = noise_path  # '/data3/mli2/mli/noise_dataset'
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.noise_prob = noise_prob

        if ('vad_MIX' or 'vad_ami_small') in manifest_path:
            self.mix_vad = True
        else:
            self.mix_vad = False
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
                    segment_span = (int(items[1]), int(items[2]))
                    self.segments.append(segment_span)

                if min_length is not None and sz < min_length:
                    skipped += 1
                    continue
                self.fnames.append(items[0])
                self.sizes.append(sz)
        self.noise_files = None
        if self.noise_path and self.task in ['scd', 'vad']:
            self.noise_files = []
            for dirpath, dirnames, filenames in os.walk(self.noise_path):
                for name in filenames:
                    self.noise_files.append(os.path.join(dirpath, name))

        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

    def __getitem__(self, index):

        fname = os.path.join(self.root_dir, self.fnames[index])
        wav, curr_sample_rate = sf.read(fname)
        if curr_sample_rate != self.sample_rate:
            wav = librosa.resample(wav, curr_sample_rate, self.sample_rate)
            curr_sample_rate = self.sample_rate

        if self.segments:
            segment_span = self.segments[index]
            start = segment_span[0]
            end = segment_span[1]
            wav = wav[start:end]

        wav_len = len(wav)
        noise_file = None
        if not self.noise_files is None and random.random() <= self.noise_prob and len(wav.shape) == 1:
            # print(111111,sys.exit())
            noise_file = random.choice(self.noise_files)
            noise, noise_sr = sf.read(noise_file)
            if len(noise.shape) > 1:
                noise = noise[:, 0]
                # print(noise.shape)
            len_noise = len(noise)
            assert noise_sr == curr_sample_rate
            while len_noise < wav_len:
                noise = np.concatenate((noise, noise), axis=0)
                len_noise = len(noise)
            rand_noise_start = random.randint(0, len_noise - wav_len)
            crop_noise = noise[rand_noise_start:rand_noise_start + wav_len]
            noise = normalize_wav(crop_noise)
            snr = (self.snr_max - self.snr_min) * np.random.random_sample() + self.snr_min
            alpha = np.exp(-np.log(10) * snr / 20)
            wav = wav + alpha * noise

        feats = torch.from_numpy(wav).float()

        feats = self.postprocess(feats, curr_sample_rate)

        if torch.any(torch.isnan(feats)):
            print('is_nan', fname)
            sys.exit()
        if torch.any(torch.isinf(feats)):
            print('isinf', fname)
            sys.exit()

        return {"id": index, "source": feats, 'task': self.task}


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
            snr_max=15,
            noise_prob=0.85,
            ts_prob=0.6,
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
        self.noise_path = noise_path  # '/data3/mli2/mli/noise_dataset'
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.noise_prob = noise_prob
        self.ts_prob = ts_prob
        self.speaker_embeddings = {}
        self.speaker_embeddings = {}
        ts_emb_path_list = ts_emb_path.split(',')
        for ts_emb_path in ts_emb_path_list:
            speakers = os.listdir(ts_emb_path)
            for speaker in speakers:
                emb_path = os.path.join(ts_emb_path, speaker, '%s.npy' % speaker)
                if os.path.exists(emb_path):
                    sp_emb = np.load(emb_path)
                    sp_emb = torch.from_numpy(sp_emb)
                    self.speaker_embeddings[speaker] = sp_emb
                    # print(111111,speaker,emb_path)

                else:
                    print('not exist', emb_path)
        skipped = 0
        # sys.exit()
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
                    segment_span = (int(items[1]), int(items[2]))
                    self.segments.append(segment_span)

                if min_length is not None and sz < min_length:
                    skipped += 1
                    continue
                self.fnames.append(items[0])
                self.sizes.append(sz)
        self.noise_files = None
        if self.noise_path and self.task in ['scd', 'vad']:
            self.noise_files = []
            for dirpath, dirnames, filenames in os.walk(self.noise_path):
                for name in filenames:
                    self.noise_files.append(os.path.join(dirpath, name))

        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

    def __getitem__(self, index):
        fname = os.path.join(self.root_dir, self.fnames[index])
        wav, curr_sample_rate = sf.read(fname)
        if curr_sample_rate != self.sample_rate:
            wav = librosa.resample(wav, curr_sample_rate, self.sample_rate)
            curr_sample_rate = self.sample_rate

        if self.segments:
            segment_span = self.segments[index]
            start = segment_span[0]
            end = segment_span[1]
            wav = wav[start:end]
        wav_len = len(wav)

        ts_label = None
        input_embedding = None
        if self.task == 'vad':
            cur_spk = self.fnames[index].split('/')[0]
            if random.random() <= self.ts_prob and cur_spk in self.speaker_embeddings.keys():
                input_embedding = self.speaker_embeddings[cur_spk]
                #ts_label = torch.tensor([0], dtype=torch.int32)
                rand_fname = random.choice(self.fnames)
                rand_audio = os.path.join(self.root_dir,rand_fname)
                rand_spk = rand_fname.split('/')[0]
                if cur_spk!=rand_spk:
                    rand_wav, curr_sample_rate = sf.read(rand_audio)
                    if curr_sample_rate != self.sample_rate:
                        rand_wav = librosa.resample(rand_wav, curr_sample_rate, self.sample_rate)
                        curr_sample_rate = self.sample_rate

                    if len(rand_wav)>self.sample_rate:
                        rand_wav = rand_wav[self.sample_rate:]
                    rand_wav_len = len(rand_wav)
                    if random.random() < 0.5:
                        ovp_len = random.randint(0, 8000)
                        tgt_len = wav_len + rand_wav_len - ovp_len
                        pad_wav = np.pad(wav, (0, tgt_len - wav_len), 'constant', constant_values=(0, 0))
                        pad_rand_wav = np.pad(rand_wav, (tgt_len - rand_wav_len, 0), 'constant', constant_values=(0, 0))
                        wav = pad_wav + pad_rand_wav

                        rand_start = random.randint(0, min(wav_len - 3200, len(wav) - self.max_sample_size))
                        rand_end = rand_start + self.max_sample_size
                        wav = wav[rand_start:rand_end]
                        assert len(wav) == self.max_sample_size
                        tgt_st = 0
                        tgt_ed = min(self.max_sample_size, wav_len - rand_start)


                    else:
                        ovp_len = random.randint(0, 8000)
                        tgt_len = wav_len + rand_wav_len - ovp_len
                        pad_wav = np.pad(wav, (tgt_len - wav_len, 0), 'constant', constant_values=(0, 0))
                        pad_rand_wav = np.pad(rand_wav, (0, tgt_len - rand_wav_len), 'constant', constant_values=(0, 0))
                        wav = pad_wav + pad_rand_wav

                        rand_end = max(self.max_sample_size, random.randint(rand_wav_len + 3200, len(wav)))
                        rand_start = rand_end - self.max_sample_size
                        wav = wav[rand_start:rand_end]
                        assert len(wav) == self.max_sample_size
                        tgt_st = max(0, rand_wav_len - rand_start - ovp_len)
                        tgt_ed = self.max_sample_size

                    tgt_seg_dict = {'start': tgt_st / self.sample_rate, 'end': tgt_ed / self.sample_rate}
                    feats_len = (len(wav) - 320) // 320
                    ts_ys = np.ones(feats_len, dtype=np.int64)

                    segment = Segment.from_json(tgt_seg_dict)
                    for i, j in sw.crop(segment, mode='loose', return_ranges=True):
                        i = max(0, i)
                        j = min(feats_len, j)
                        ts_ys[i:j] = 0

                    ts_label = torch.from_numpy(ts_ys)
                    assert ts_label.size(0)==100

            elif cur_spk not in self.speaker_embeddings.keys():
                random_spk = random.choice(list(self.speaker_embeddings.keys()))
                input_embedding = self.speaker_embeddings[random_spk]
                ts_label = torch.tensor([2], dtype=torch.int32)
                feats_len = (self.max_sample_size- 320) // 320
                ts_label = ts_label.expand(feats_len)
            else:
                random_spk = random.choice(list(self.speaker_embeddings.keys()))
                input_embedding = self.speaker_embeddings[random_spk]
                if random_spk == cur_spk:
                    ts_label = torch.tensor([0], dtype=torch.int32)
                else:
                    ts_label = torch.tensor([1], dtype=torch.int32)
                feats_len = (self.max_sample_size - 320) // 320
                ts_label = ts_label.expand(feats_len)
            #print(11111,ts_label)
            #sys.exit()

        # noise data augmentation
        if not self.noise_files is None and random.random() <= self.noise_prob:
            noise_file = random.choice(self.noise_files)
            noise, noise_sr = sf.read(noise_file)
            if len(noise.shape) > 1:
                noise = noise.mean(axis=1)
                # print(noise.shape)
            len_noise = len(noise)
            assert noise_sr == curr_sample_rate
            while len_noise < wav_len:
                noise = np.concatenate((noise, noise), axis=0)
                len_noise = len(noise)
            rand_noise_start = random.randint(0, len_noise - wav_len)
            crop_noise = noise[rand_noise_start:rand_noise_start + wav_len]
            noise = normalize_wav(crop_noise)
            snr = (self.snr_max - self.snr_min) * np.random.random_sample() + self.snr_min
            alpha = np.exp(-np.log(10) * snr / 20)
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)
            wav = wav + alpha * noise

        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)


        return {"id": index, "source": feats, 'task': self.task, 'input_embedding': input_embedding,
                'ts_label': ts_label}

