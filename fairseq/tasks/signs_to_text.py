# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from argparse import Namespace
import json

import pandas as pd
import numpy as np

from fairseq.data import AddTargetDataset, Dictionary, encoders

from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import ChoiceEnum
from typing import Optional, Any
from omegaconf import MISSING, II, DictConfig

from fairseq.data.sign_language import (
    SignFeatsType,
    SignFeatsDataset,
)
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

from fairseq import metrics, search, tokenizer, utils
import pdb 

logger = logging.getLogger(__name__)

EVAL_BLEU_ORDER = 4

@dataclass
class SignsToTextConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    max_source_positions: Optional[int] = field(
        default=750, metadata={"help": "max number of tokens in the source sequence"}
    )
    min_source_positions: Optional[int] = field(
        default=50, metadata={"help": "min number of tokens in the source sequence"}
    )
    max_target_positions: Optional[int] = field(
        default=512, metadata={"help": "max number of tokens in the target sequence"}
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    body_parts: str = field(
        default = "face,upperbody,lefthand,righthand",
        metadata={"help": "Select the keypoints that you want to use. Options: 'face','upperbody','lowerbody','lefthand', 'righthand'"},
    )
    feat_dims: str = field(
        default = "0,1,2",
        metadata={"help": "Select the keypoints dimensions that you want to use. Options: 0, 1, 2, 3"},
    )
    tokenizer_type: str = field(
        default='sentencepiece',
        metadata={"help": "subword tokenizer type"},
    )
    tokenizer_vocab: str = field(
        default=MISSING,
        metadata={"help": "subword tokenizer file"},
    )
    shuffle_dataset: bool = field(
        default=True,
        metadata={"help": "set True to shuffle the dataset between epochs"},
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={"help": "number of buckets"},
    )
    text_compression_level: ChoiceEnum([x.name for x in TextCompressionLevel]) = field(
        default="none",
        metadata={
            "help": "compression level for texts (e.g. audio filenames, "
            "target texts): none/low/high (default: none). "
        },
    )
    feats_type: ChoiceEnum([x.name for x in SignFeatsType]) = field(
        default="keypoints",
        metadata={
            "help": "type of features for the sign input data: keypoints/i3d (default: keypoints). "
        },
    )
    tpu: bool = II("common.tpu")
    bpe_sentencepiece_model: str = II("bpe.sentencepiece_model")

    #add the following for reporting BLEU during validation
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_args: Optional[str] = field(
        default="{}",
        metadata={
            "help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_print_samples: bool = field(
        default=True, metadata={"help": "print sample generations during validation"}
    )

@register_task("signs_to_text", dataclass=SignsToTextConfig)
class SignsToTextTask(FairseqTask):
    def __init__(self, cfg, tgt_dict):
        super().__init__(cfg)
        self.tgt_dict = tgt_dict
        self.bpe_tokenizer = self.build_bpe(
            Namespace(
                bpe='sentencepiece',
                sentencepiece_model=cfg.bpe_sentencepiece_model
            )
        )

    @classmethod
    def setup_task(cls, cfg):
        dict_path = Path(cfg.bpe_sentencepiece_model).with_suffix('.txt')
        if not dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
        tgt_dict = Dictionary.load(dict_path.as_posix())
        logger.info(
            f"dictionary size ({dict_path.name}): " f"{len(tgt_dict):,}"
        )
        return cls(cfg, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):

        root_dir = Path(self.cfg.data)
        assert root_dir.is_dir(), f"{root_dir} does not exist"

        # TODO: Change when we add i3d features
        manifest_file = root_dir / f"{split}_filt.tsv"
        if SignFeatsType(self.cfg.feats_type) == SignFeatsType.keypoints:
            feats_file = root_dir / f"{split}_sent.h5"
        elif SignFeatsType(self.cfg.feats_type) == SignFeatsType.i3d:
            if split =='train':
                manifest_file = root_dir / f"{split}_filt_i3d.tsv" #remove the ones not in the h5 file
            feats_file = root_dir / f"{split}_i3d.h5" #Check if this is at sentence level, because I don't think so...
        elif SignFeatsType(self.cfg.feats_type) == SignFeatsType.CNN2d:
            feats_file = root_dir / f'{split}_sent.h5'
        else:
            raise NotImplementedError("Features other than CNN2d, i3d or keypoints are not implemented")

        if self.cfg.num_batch_buckets > 0 or self.cfg.tpu:
            raise NotImplementedError("Pending to implement bucket_pad_length_dataset wrapper")

        self.datasets[split] = SignFeatsDataset.from_manifest_file(
            manifest_file=manifest_file,
            feats_file=feats_file,
            feats_type=self.cfg.feats_type,
            bodyparts=self.cfg.body_parts.split(','),
            feat_dims=[int(d) for d in self.cfg.feat_dims.split(',')],
            min_sample_size=self.cfg.min_source_positions,
            max_sample_size=self.cfg.max_source_positions,
            shuffle=self.cfg.shuffle_dataset,
            normalize=self.cfg.normalize,
            text_compression_level=self.cfg.text_compression_level,
        )

        data = pd.read_csv(manifest_file, sep="\t")
        text_compressor = TextCompressor(level=self.cfg.text_compression_level)

        labels = [
            text_compressor.compress(row['SENTENCE']) #added this
            for i, row in data.iterrows()
            if row['SENTENCE_NAME'] not in self.datasets[split].skipped_ids
        ]

        assert len(labels) == len(self.datasets[split]), (
            f"labels length ({len(labels)}) and dataset length "
            f"supposed to skip ({len(self.datasets[split].skipped_ids)}) ids"
            f"({len(self.datasets[split])}) do not match"
        )

        def process_label_fn(label):
            return self.target_dictionary.encode_line(
                self.bpe_tokenizer.encode(label), append_eos=False, add_if_not_exist=False
            )

        def label_len_fn(label):
            return len(self.bpe_tokenizer.encode(label))

        self.datasets[split] = AddTargetDataset(
            self.datasets[split],
            labels,
            pad=self.target_dictionary.pad(),
            eos=self.target_dictionary.eos(),
            batch_targets=True,
            process_label=process_label_fn, 
            label_len_fn=label_len_fn,
            add_to_input=True,
            text_compression_level=self.cfg.text_compression_level,
        )        

    #Add this for validation
    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg)
        if self.cfg.eval_bleu:
            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    #Add this for validation
    def valid_step(self, sample, model, criterion): 
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_bleu: 
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.cfg.max_source_positions, self.cfg.max_target_positions

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = []
        for l in lines:
            h5_file, _id = l.split(':')
            feats_file = h5py.File(h5_file, "r")
            n_frames.append(np.array(feats_file[_id]).shape[0])

        return lines, n_frames

    # TODO: Implement this  method
    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        raise NotImplementedError
        return SpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )
    #Add this for validation
    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_bleu:

            def sum_logs(key):
                import torch

                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect

                    try:
                        from sacrebleu.metrics import BLEU

                        comp_bleu = BLEU.compute_bleu
                    except ImportError:
                        # compatibility API for sacrebleu 1.x
                        import sacrebleu

                        comp_bleu = sacrebleu.compute_bleu

                    fn_sig = inspect.getfullargspec(comp_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = comp_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth,
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu
        #breakpoint()
        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.bpe_tokenizer:
                s = self.bpe_tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])