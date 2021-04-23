# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import BaseWrapperDataset
from . import data_utils
import sys

class AddTargetDataset(BaseWrapperDataset):
    def __init__(self, dataset, labels, pad, eos, batch_targets, process_label=None, add_to_input=False):
        super().__init__(dataset)
        self.labels = labels
        self.batch_targets = batch_targets
        self.pad = pad
        self.eos = eos
        self.process_label = process_label
        self.add_to_input = add_to_input

    def get_label(self, index):
        return self.labels[index] if self.process_label is None else self.process_label(self.labels[index])

    def __getitem__(self, index):
        item = self.dataset[index]
        #_id = item['id']
        #item["label"] = self.get_label(_id)
        item["label"] = self.get_label(index)
        #print(5555555,_id,item,self.labels[_id])
        #sys.exit()
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = len(self.get_label(index))
        return (sz, own_sz)

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        target = [s["label"] for s in samples if s["id"] in indices]
        if self.batch_targets:
            collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
            target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
        collated["target"] = target
        if self.add_to_input:
            eos = target.new_full((target.size(0), 1), self.eos)
            collated["target"] = torch.cat([target, eos], dim=-1)
            collated["net_input"]["prev_output_tokens"] = torch.cat([eos, target], dim=-1)
        return collated


class AddTargetDataset_ts(BaseWrapperDataset):
    def __init__(self, dataset, labels, pad, eos, batch_targets, process_label=None, add_to_input=False):
        super().__init__(dataset)
        self.labels = labels
        self.batch_targets = batch_targets
        self.pad = pad
        self.eos = eos
        self.process_label = process_label
        self.add_to_input = add_to_input


    def get_label(self, index):
        return self.labels[index] if self.process_label is None else self.process_label(self.labels[index])

    def __getitem__(self, index):
        item = self.dataset[index]
        #_id = item['id']
        #item["label"] = self.get_label(_id)
        #item["label"] = self.get_label(index)
        if item['task']=='vad':
            item["label"] = self.get_label(index)
            item["ts_label"] = item['ts_label']
        else:
            item["label"] = self.get_label(index)

        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = len(self.get_label(index))
        return (sz, own_sz)

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        input_embeddings = [s["input_embedding"] for s in samples]
        input_embeddings = torch.stack(input_embeddings,dim=0)
        collated["net_input"]["input_embeddings"] = input_embeddings

        target = [s["label"] for s in samples if s["id"] in indices]
        target_ts = [s["ts_label"] for s in samples if s["id"] in indices]

        if self.batch_targets:
            collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
            target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
            target_ts = data_utils.collate_tokens(target_ts, pad_idx=self.pad, left_pad=False)

        collated["target"] = target
        collated["target_ts"] = target_ts
        if self.add_to_input:
            eos = target.new_full((target.size(0), 1), self.eos)
            collated["target"] = torch.cat([target, eos], dim=-1)
            collated["net_input"]["prev_output_tokens"] = torch.cat([eos, target], dim=-1)
        return collated