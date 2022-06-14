#!/usr/bin/env python3

# This code is based on the speech_to_text implementation (commit: d974c70)
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import h5py
import argparse
import logging
import pandas as pd
from typing import Tuple
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch.utils.data import Dataset
import pdb
from examples.speech_to_text.data_utils import (
    gen_vocab,
    save_df_to_tsv,
)

from utils import h5_video2sentence

log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "signs", "n_frames", "tgt_text"]


class How2Sign(Dataset):
    """
    Create a Dataset for How2Sign. Each item is a tuple of the form:
    signs, target sentence
    """

    LANGUAGES = ["en"] # TODO: add "pt"
    SPLITS = ["train", "val", "test"]

    def __init__(self, root: str, lang: str, split: str) -> None:
        self.root = Path(root)
        assert split in self.SPLITS and lang in self.LANGUAGES
        assert self.root.is_dir()
        self.h5_sign = h5py.File(self.root / f"{split}_sent.h5", "r")

        self.data = pd.read_csv(self.root / f"{split}.tsv", sep="\t")
        for i, row in self.data.iterrows():
            #not finding anything in here.... why??
            #pdb.set_trace()
            if row['SENTENCE_NAME'] not in list(self.h5_sign.keys()):
                print(f"Error with keypoint {row['SENTENCE_NAME']}, not found inside h5_sign") 
                self.data.drop(i, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, str, str]:
        sent_id = self.data.loc[n, 'SENTENCE_NAME']
        src_signs = torch.Tensor(self.h5_sign[sent_id])
        tgt_sent = self.data.loc[n, 'SENTENCE']
        return sent_id, src_signs, tgt_sent

    def __len__(self) -> int:
        return len(self.data)

    def filter_by_length(self, min_n_frames: int, max_n_frames: int) -> None:
        lengths = self.data['END_FRAME'] - self.data['START_FRAME'] + 1
        self.data = self.data[lengths.between(min_n_frames, max_n_frames)]


def process(args):
    root = Path(args.data_root).absolute()

    for split in How2Sign.SPLITS:
        print(f"Processing '{split}' split")
        input_tsv = root / f"{split}.tsv"
        filt_tsv = root / f"{split}_filt.tsv"
        if args.data_type == 'skeletons':
            signs_video = root / f"{split}.h5"
            signs_sentence = root / f"{split}_sent.h5"
        elif args.data_type == 'i3d':
            signs_video = root / f"{split}_i3d.h5"
            signs_sentence = root / f"{split}_i3d_sent.h5"
        else:
            print('Error with data_type, not i3d or skeletons')
        try:
            h5_video2sentence(input_tsv, signs_video, signs_sentence, overwrite=args.overwrite)
        except FileNotFoundError:
            print(f"Skipping '{split}' split")
            continue
        except FileExistsError:
            print(f"Reusing sentence-level h5 for '{split}' split. Set --overwrite to overwrite it.")
        print(f'signs_video: {signs_video}, signs_sentence: {signs_sentence}')
        for lang in How2Sign.LANGUAGES:
            dataset = How2Sign(root, lang, split)
            
            if split == 'train':
                '''
                print(f"Generating vocab for '{lang}' language")
                v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
                spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{lang}"
                with NamedTemporaryFile(mode="w") as f:
                    for i in range(len(dataset)):
                        f.write(dataset[i][2] + "\n")
                    gen_vocab(
                        Path(f.name),
                        root / spm_filename_prefix,
                        args.vocab_type,
                        args.vocab_size,
                    )
                '''
                print("Filtering samples by length...")
                dataset.filter_by_length(args.min_n_frames, args.max_n_frames)
                print(f"{len(dataset)} samples after filtering")

            print("Saving dataframe...")
            save_df_to_tsv(dataset.data, filt_tsv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--min-n-frames", default=5, type=int)
    parser.add_argument("--max-n-frames", default=1000, type=int)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        type=str,
        choices=["bpe", "unigram", "char"],
    )
    parser.add_argument("--data-type", default='skeletons', type=str)
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
