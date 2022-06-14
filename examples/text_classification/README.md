# Topic Detection on How2Sign text data

Inspired on [this tutorial](https://github.com/pytorch/fairseq/blob/main/examples/roberta/README.custom_classification.md).

1. Obtain data and put it in `../../../data/How2Sign/text/`. You need to include `{split}.csv`, `{split}.txt` and `{split}.h5`.

2. Launch `prep_how2sign.py` in order to preprocess the How2Sign data.

3. Move file `format_data.py` to directory `../../../data/How2Sign/text/`. From within said directory, run `format_data.py` to give data the appropiate format.

4. Back in directory `/examples/text_classification`, run `bpe_encode.sh` to download and use the vocabulary encoder.

5. Execute script `preprocess_data.sh` so that data is prepared according to RoBERTa's specifications.

6. To run training, run script `launch_run_training.sh`. Note that prior to that, you must have downloaded the pretrained weights for RoBERTa base from [this webpage](https://github.com/pytorch/fairseq/tree/main/examples/roberta#pre-trained-models) inside directory `../../../data/How2Sign/text/`.
