# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import OrderedDict
import os
import sys
import itertools
import logging
import os
from fairseq.data.multi_corpus_sampled_dataset import MultiCorpusSampledDataset
from fairseq.data.multi_corpus_dataset import MultiCorpusDataset

import numpy as np

from fairseq.data import FileAudioDataset_ts, Dictionary, AddTargetDataset_ts,MTL_Dictionary,FileAudioDataset,AddTargetDataset
from . import FairseqTask, register_task

logger = logging.getLogger(__name__)

class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


@register_task("audio_mtl_ts")
class Audio_MTL_PredictionTask_ts(FairseqTask):
    """

    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="path to data directory")
        parser.add_argument(
            "--sample-rate",
            default=16000,
            type=int,
            help="target sample rate. audio files will be up/down sampled to this rate",
        )
        parser.add_argument(
            "--normalize",
            action="store_true",
            help="if set, normalizes input to have 0 mean and unit variance",
        )
        parser.add_argument(
            "--max-sample-size",
            default=None,
            type=int,
            help="max sample size to crop to for batching. default = min sample length",
        )
        parser.add_argument(
            "--min-sample-size",
            default=None,
            type=int,
            help="min sample size to crop to for batching. default = same as --max-sample-size",
        )

        parser.add_argument(
            "--enable-padding",
            action="store_true",
            help="pad shorter samples instead of cropping",
        )

        parser.add_argument(
            "--no-min-cropping", action="store_true", help="always crop to max sample size or smallest length"
        )

        parser.add_argument(
            "--labels",
            type=str,
            default=None,
            help="extension of the label file to load, if any",
        )

        parser.add_argument(
            "--datasets",
            type=str,
            default=None,
            help="datasets for tasks for example: train_AP18,train_vox",
        )

        parser.add_argument(
            "--datasets_dict",
            type=str,
            default=None,
            help="datasets for tasks for example: train_AP18,train_vox",
        )
        parser.add_argument(
            "--noise-path",
            type=str,
            default=None,
            help="extension of the noise files to load, if any",
        )
        parser.add_argument(
            "--ts-path",
            type=str,
            default=None,
            help="extension of the target embeddings to load, if any",
        )
    def __init__(self, args, source_dictionary=None):
        super().__init__(args)
        self._target_dictionary = None
        self._source_dictionary = source_dictionary
        self.is_ctc = args.criterion == "ctc"
        self._target_dictionaries = {}


        if self.args.datasets_dict:
            datasets = self.args.datasets_dict.split(',')
            # datasets = ["lid_AP18","sid_vox1","vad_hkust","asr_hkust"] # "lid_AP18","sid_vox1"
            dict_paths = [os.path.join('examples/wav2vec/manifest/MTL', subset, f"dict.cls.txt") for subset in
                          datasets]
            self._target_dictionary = Dictionary.load_files(dict_paths)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        if not self.args.datasets:
            manifest = os.path.join(self.args.data, "{}.tsv".format(split))
            if split == 'train':
                noise_path = self.args.noise_path
            else:
                noise_path = None
            self.datasets[split] = FileAudioDataset(
                manifest,
                sample_rate=self.args.sample_rate,
                max_sample_size=self.args.max_sample_size,
                min_sample_size=self.args.min_sample_size if not self.args.no_min_cropping else self.args.max_sample_size,
                min_length=self.args.min_sample_size,
                pad=self.args.labels is not None or self.args.enable_padding,
                normalize=self.args.normalize,
                noise_path=noise_path
            )

            if self.args.labels:
                datasets = self.args.datasets_dict.split(',')
                dict_paths = [os.path.join('examples/wav2vec/manifest/MTL', subset, f"dict.cls.txt") for subset in
                              datasets]
                self._target_dictionary = Dictionary.load_files(dict_paths)

                label_path = os.path.join(self.args.data, f"{split}.{self.args.labels}")
                labels = []
                with open(label_path, "r") as f:
                    for line in f:
                        labels.append(line)

                process_label = LabelEncoder(self.target_dictionary)

                self.datasets[split] = AddTargetDataset(
                    self.datasets[split],
                    labels,
                    pad=self.target_dictionary.pad(),
                    eos=self.target_dictionary.eos(),
                    batch_targets=True,
                    process_label=process_label,
                    add_to_input=not self.is_ctc,
                )
        else:
            datasets = self.args.datasets.split(',')
            dataset_map = OrderedDict()
            if self.args.labels:
                dict_paths = [os.path.join(self.args.data,subset, f"dict.{self.args.labels}.txt") for subset in datasets]
                self._target_dictionary = Dictionary.load_files(dict_paths)

            '''' modified by mli
            for subset in datasets:
                task = subset.split('_')[0]
                task_dict_path = os.path.join(self.args.data,subset, f"dict.{self.args.labels}.txt")
                task_dictionary = Dictionary.load(task_dict_path)
                self._target_dictionaries[task] = task_dictionary
            '''

            for dataset in datasets:
                manifest = os.path.join(self.args.data,dataset, "{}.tsv".format(split))
                if split == 'train':
                    noise_path = self.args.noise_path
                else:
                    noise_path = None

                if 'asr' in dataset:
                    max_sample_size = None
                    min_sample_size = self.args.min_sample_size
                else:
                    max_sample_size = self.args.max_sample_size
                    min_sample_size = None


                dataset_map[dataset] = FileAudioDataset_ts(
                    manifest,
                    self.args.ts_path,
                    sample_rate=self.args.sample_rate,
                    max_sample_size=max_sample_size,
                    min_sample_size=min_sample_size if not self.args.no_min_cropping else max_sample_size,
                    min_length=min_sample_size,
                    pad=self.args.labels is not None or self.args.enable_padding,
                    normalize=self.args.normalize,
                    noise_path=noise_path
                )

                if self.args.labels:
                    label_path = os.path.join(self.args.data,dataset, f"{split}.{self.args.labels}.txt")
                    labels = []
                    with open(label_path, "r") as f:
                        for line in f:
                            labels.append(line)

                    process_label = LabelEncoder(self.target_dictionary)

                    dataset_map[dataset] = AddTargetDataset_ts(
                        dataset_map[dataset],
                        labels,
                        pad=self.target_dictionary.pad(),
                        eos=self.target_dictionary.eos(),
                        batch_targets=True,
                        process_label=process_label,
                        add_to_input=False,
                    )

            #self.datasets[split] = MultiCorpusDataset(dataset_map,distribution=[0.5,0.5],seed=1234)
            self.datasets[split] = MultiCorpusSampledDataset(dataset_map)
            logger.info('{} {} examples'.format(
                split, len(self.datasets[split]))
            )



    @property
    def source_dictionary(self):
        return self._source_dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self._target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (sys.maxsize, sys.maxsize)
