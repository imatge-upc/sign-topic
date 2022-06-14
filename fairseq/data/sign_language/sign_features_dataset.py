# This code is inspired by the raw_audio_dataset implementation (commit: 1575f30)
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import logging
from enum import Enum
from pathlib import Path
from typing import List, Union, Optional

import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import pdb

from fairseq.data import FairseqDataset, BaseWrapperDataset, RandomCropDataset
from fairseq.data.data_utils import (
    compute_mask_indices,
    numpy_seed
)

from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

logger = logging.getLogger(__name__)

class SignFeatsType(Enum):
    keypoints = "keypoints"
    i3d = "i3d"
    CNN2d = "CNN2d"

class SignFeatsDataset(FairseqDataset):
    def __init__(
        self,
        ids: List[str],
        feats_file: Union[Path, str],
        sizes: List[int] = None,
        feats_type: SignFeatsType = SignFeatsType.keypoints,
        bodyparts: Optional[List[str]] = None,
        feat_dims: List[int] = [0, 1, 2, 3],
        min_sample_size: int = 0,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        normalize: bool = False,
        text_compression_level: TextCompressionLevel = TextCompressionLevel.none,
    ):
        super().__init__()

        self.text_compressor = TextCompressor(level=text_compression_level)

        self.ids = [self.text_compressor.compress(_id) for _id in ids]
        self.feats_file = h5py.File(feats_file, "r") # XXX: This might be a problem, check later

        if sizes is None:
            sizes = []
            for _id in self.ids:
                _id = self.text_compressor.decompress(_id)
                sizes.append(np.array(self.feats_file[_id]).shape[0])

        self.sizes = sizes

        self.feats_type = feats_type
        self.bodyparts = bodyparts
        self.feat_dims = feat_dims

        self.shuffle = shuffle
        self.normalize = normalize

        self.min_sample_size = min_sample_size
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.skipped_ids = []
        for _id, size in zip(self.ids[:], self.sizes[:]):
            if size < self.min_sample_size or size > self.max_sample_size:
                self.sizes.pop(self.ids.index(_id))
                self.ids.remove(_id)
                self.skipped_ids.append(self.text_compressor.decompress(_id))
        logger.info(f"Skipped {len(self.skipped_ids)} sentences, that were too short or too long.")

        try:
            import pyarrow as pa
            self.ids = pa.array(self.ids)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

    @staticmethod
    def list_avail_ids(feats_file: Union[Path, str]):
        feats_file = h5py.File(feats_file, "r")
        return list(feats_file.keys())

    @classmethod
    def from_manifest_file(cls, manifest_file: Union[str, Path], **kwargs):
        ids = []
        sizes = []
        manifest = pd.read_csv(manifest_file, sep="\t")
        for _, row in manifest.iterrows():
            ids.append(row['SENTENCE_NAME'])
            size = int(row['END_FRAME']) - int(row['START_FRAME'])
            sizes.append(size)
        logger.info(f"loaded {len(ids)} samples")

        return cls(ids, sizes=sizes, **kwargs)

    def __getitem__(self, index):
        _id = self.ids[index]
        _id = _id if isinstance(self.ids, list) else _id.as_py()
        fn = self.text_compressor.decompress(_id)
        feats = torch.Tensor(np.array(self.feats_file[fn])).float()
        feats = self.postprocess(feats)

        return {"id": index, "h2s_id": _id, "source": feats}

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats):
        if SignFeatsType[self.feats_type] is SignFeatsType.keypoints: #added SignFeatsType[] to be able to compare
            from fairseq.data.sign_language.utils import (
                select_keypoints_by_bodypart,
                select_keypoints_by_dimension,
            ) # FIXME: check how to do this imports better
            feats, n_feats = select_keypoints_by_bodypart(feats, self.bodyparts)
            feats = select_keypoints_by_dimension(feats, self.feat_dims)
            feats_split = feats.reshape(-1, n_feats, 3).permute(2, 0, 1)
            with torch.no_grad():
                feats_norm_split = F.layer_norm(feats_split, feats_split.shape[1:])
            feats = feats_norm_split.permute(1, 2, 0).reshape(-1, n_feats * 3).contiguous()
        elif SignFeatsType[self.feats_type] is SignFeatsType.i3d or SignFeatsType[self.feats_type] is SignFeatsType.CNN2d:
            # should we actually normalize CNN2d features?
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape) #check this normalization
        else:
            raise NotImplementedError(f"Using {self.feats_type} which is not SignFeatsType.keypoints or SignFeatsType.i3d or SignFeatsType.2dCNN")
        return feats


    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        collated_sources = sources[0].new_zeros(len(sources), max(sizes), sources[0].shape[-1])

        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - max(sizes)
            collated_sources[i] = torch.cat(
                [source, source.new_full((-diff, source.shape[-1]), 0.0)]
            )

        return {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                "src_tokens": collated_sources, 
                "src_lengths": torch.Tensor(sizes) # FIXME: If you use buckets
            }
        }

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        if self.shuffle:
            order = np.lexsort(
                [np.random.permutation(len(self)), np.array(self.sizes)]
            )
            return order[::-1]
        else:
            return np.arange(len(self))


# TODO: In task, if compute_mask_indices=True, create dataset of this type
# TODO: In task, if using this, it may be useful to wrap it also with RandomCropSignFeatsDataset (remember paddings)
class MaskSignFeatsDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: SignFeatsDataset,
        **mask_compute_kwargs,
        ):
        super().__init__(dataset)
        self.mask_compute_kwargs = mask_compute_kwargs
        self._features_size_map = {}
        self._C = mask_compute_kwargs["encoder_embed_dim"]
        self._conv_feature_layers = eval(mask_compute_kwargs["conv_feature_layers"])

    def _compute_mask_indices(self, dims, padding_mask):
        # Create masks for Sign2vec pretraining
        raise NotImplementedError("This feature is still not available")
        B, T, C = dims
        mask_indices, mask_channel_indices = None, None
        if self.mask_compute_kwargs["mask_prob"] > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_compute_kwargs["mask_prob"],
                self.mask_compute_kwargs["mask_length"],
                self.mask_compute_kwargs["mask_selection"],
                self.mask_compute_kwargs["mask_other"],
                min_masks=2,
                no_overlap=self.mask_compute_kwargs["no_mask_overlap"],
                min_space=self.mask_compute_kwargs["mask_min_space"],
            )
            mask_indices = torch.from_numpy(mask_indices)
        if self.mask_compute_kwargs["mask_channel_prob"] > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_compute_kwargs["mask_channel_prob"],
                self.mask_compute_kwargs["mask_channel_length"],
                self.mask_compute_kwargs["mask_channel_selection"],
                self.mask_compute_kwargs["mask_channel_other"],
                no_overlap=self.mask_compute_kwargs["no_mask_channel_overlap"],
                min_space=self.mask_compute_kwargs["mask_channel_min_space"],
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices).unsqueeze(1).expand(-1, T, -1)
            )

        return mask_indices, mask_channel_indices

    def _get_mask_indices_dims(self, size, padding=0, dilation=1):
        raise NotImplementedError("This feature is still not available")
        if size not in self._features_size_map:
            L_in = size
            for (_, kernel_size, stride) in self._conv_feature_layers:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            self._features_size_map[size] = L_out
        return self._features_size_map[size]

    def collater(self, samples):
        out = self.dataset.collater(samples)
        raise NotImplementedError("This feature is still not available")

        B = out["net_input"]["source"].size(0)
        T = self._get_mask_indices_dims(out["net_input"]["source"].size(-2))
        padding_mask_reshaped = out["net_input"]["padding_mask"].clone()
        extra = padding_mask_reshaped.size(1) % T
        if extra > 0:
            padding_mask_reshaped = padding_mask_reshaped[:, :-extra]
        padding_mask_reshaped = padding_mask_reshaped.view(
            padding_mask_reshaped.size(0), T, -1
        )
        padding_mask_reshaped = padding_mask_reshaped.all(-1)
        out["net_input"]["padding_count"] = padding_mask_reshaped.sum(-1).max().item()
        mask_indices, mask_channel_indices = self._compute_mask_indices(
            (B, T, self._C),
            padding_mask_reshaped,
        )
        out["net_input"]["mask_indices"] = mask_indices
        out["net_input"]["mask_channel_indices"] = mask_channel_indices
        out["sample_size"] = mask_indices.sum().item()
        
        return out


class RandomCropSignFeatsDataset(RandomCropDataset):
    def __init__(
        self,
        dataset: SignFeatsDataset,
        truncation_length: int,
        **kwargs,
    ):
        super().__init__(dataset, truncation_length, **kwargs)

    def __getitem__(self, index):
        with numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            item_len = item["source"].size(0)
            excess = item_len - self.truncation_length
            if excess > 0:
                start_idx = np.random.randint(0, excess)
                item["source"] = item["source"][start_idx : start_idx + self.truncation_length]
            return item
