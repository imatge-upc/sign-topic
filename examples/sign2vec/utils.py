import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import pdb

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
            print(f"Error with keypoints {r['VIDEO_NAME']}") #The error is here, why???
            continue
        print(f'arr_vid.shape: {arr_vid.shape}')
        arr_sent = arr_vid[r["START_FRAME"]:r["END_FRAME"]+1]
        h5_sent.create_dataset(r["SENTENCE_NAME"], data=arr_sent)

    h5_video.close()
    h5_sent.close()
