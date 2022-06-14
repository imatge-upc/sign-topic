import re
from typing import Dict
from typing import List
from typing import Union
from pathlib import Path
import pdb

import csv
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def save_df_to_tsv(dataframe, path: Union[str, Path]):
    _path = path if isinstance(path, str) else path.as_posix()
    dataframe.to_csv(
        _path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )


def h5_video2sentence(input_tsv: Path, input_h5: Path, output_h5: Path, overwrite=False):
    if not input_tsv.is_file():
        raise FileNotFoundError(f"{input_tsv} not found")
    if not input_h5.is_file():
        raise FileNotFoundError(f"{input_h5} not found")
    if output_h5.is_file() and not overwrite:
        raise FileExistsError(f"{output_h5} exists. Remove it or set overwrite=True")

    df = pd.read_csv(input_tsv, sep='\t')

    h5_video = h5py.File(input_h5, 'r')
    h5_sent = h5py.File(output_h5, 'w')

    for _, r in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            pdb.set_trace()
            arr_vid = np.array(h5_video[r["VIDEO_NAME"]])
        except KeyError:
            print(f"Error with keypoints {r['VIDEO_NAME']}")  # FIXME: The error is here, why???
            continue
        arr_sent = arr_vid[r["START_FRAME"]:r["END_FRAME"]+1]
        h5_sent.create_dataset(r["VIDEO_NAME"], data=arr_sent)

    h5_video.close()
    h5_sent.close()


def natural_keys(text: str):
    '''
    Used for sorting strings based on natural order.
    Alphanumerical ordering: 1, 10, 11, 2, 21...
    Natural ordering: 1, 2, 10, 11, 21...
    '''
    def atof(text):
        try:
            retval = float(text)
        except ValueError:
            retval = text
        return retval
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]


def _groupByClip(dict_text: Dict[str, str]):
    sentence_ids = list(dict_text.keys())
    sentence_ids.sort(key=natural_keys)

    dict_text_video = {}
    for utt_id in sentence_ids:
            if utt_id[:11] in dict_text_video:
                dict_text_video[utt_id[:11]] += dict_text[utt_id].replace('\n', ' ')
            else:
                dict_text_video[utt_id[:11]] = dict_text[utt_id].replace('\n', ' ')
    return dict_text_video


def load_text(file_path: str, ids: List[str], groupByClip: bool = True):
    dict_text = {}
    with open(file_path) as f:
        for line in f:
            id, text = line.split(' ', 1)  # first space separates id from text
            if id[:11] in ids:
                dict_text[id] = text

    if groupByClip:
        dict_text = _groupByClip(dict_text)

    return dict_text