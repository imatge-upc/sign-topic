#!/usr/bin/env python3

# This code is based on the speech_to_text implementation (commit: d974c70)
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import errno
import os
import h5py
import argparse
import logging
import pandas as pd
from typing import Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset
from examples.SL_topic_detection.utils import (
    save_df_to_tsv,
    load_text,
)

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ['id', 'signs', 'n_frames', 'tgt_text']


class How2Sign(Dataset):
    '''
    Create a Dataset for How2Sign.
    '''

    LANGUAGES = ['en'] # TODO: add 'pt'
    SPLITS = ['train', 'val', 'test']

    def __init__(
        self,
        root: str,
        lang: str,
        split: str
    ) -> None:
        self.root = Path(root)
        assert split in self.SPLITS and lang in self.LANGUAGES
        assert self.root.is_dir()

        try:
            self.h5_sign = h5py.File(self.root / f'{split}.h5', 'r')
        except:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), self.root / f'{split}.h5'
            )

        with h5py.File(self.root / f'{split}_filt.h5', 'w') as f:
            for key in self.h5_sign.keys():
                try:
                    f[key[:11]] = self.h5_sign[key][()]
                except:
                    pass

        self.h5_sign.close()
        self.h5_sign = h5py.File(self.root / f'{split}_filt.h5', 'r')

        self.text = load_text(self.root / f'{split}.txt', list(self.h5_sign.keys()))

        self.data = pd.read_csv(self.root / f'{split}.csv')

        self.data['TEXT'] = pd.NaT
        self.data['START_FRAME'] = pd.NaT
        self.data['END_FRAME'] = pd.NaT
        for i, row in self.data.iterrows():
            if row['VIDEO_ID'] not in list(self.h5_sign.keys()):
                print(f'Error with keypoint {row["VIDEO_ID"]}, not found inside h5_sign')
                self.data.drop(i, inplace=True)
            else:
                self.data.loc[i, 'START_FRAME'] = 0
                self.data.loc[i, 'END_FRAME'] = torch.Tensor(self.h5_sign[row['VIDEO_ID']]).shape[0]
                self.data.loc[i, 'TEXT'] = self.text[row['VIDEO_ID']]

        self.data.reset_index(drop=True, inplace=True)

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, str, str]:
        sent_id = self.data.loc[n, 'VIDEO_ID']
        src_signs = torch.Tensor(self.h5_sign[sent_id])
        text = self.data.loc[n, 'TEXT']
        categ = self.data.loc[n, 'CATEGORY']
        return sent_id, src_signs, text, categ

    def __len__(self) -> int:
        return len(self.data)

    def filter_by_length(self, min_n_frames: int, max_n_frames: int) -> None:
        lengths = self.data['END_FRAME'] - self.data['START_FRAME'] + 1
        self.data = self.data[lengths.between(min_n_frames, max_n_frames)]


def process(args):
    root = Path(args.data_root).absolute()

    for split in How2Sign.SPLITS:
        print(f'Processing "{split}" split')
        filt_csv = root / f'{split}_filt.csv'
        for lang in How2Sign.LANGUAGES:
            dataset = How2Sign(root, lang, split)

            print('Filtering samples by length...')
            dataset.filter_by_length(args.min_n_frames, args.max_n_frames)
            print(f'{len(dataset)} samples after filtering')

            print('Saving dataframe...')
            save_df_to_tsv(dataset.data, filt_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', '-d', required=True, type=str)
    parser.add_argument('--min-n-frames', default=150, type=int)
    parser.add_argument('--max-n-frames', default=5500, type=int)
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()

    process(args)


if __name__ == '__main__':
    main()
