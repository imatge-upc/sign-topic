# Download fairseq dictionary.
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

H2S_ROOT=../../../data/How2Sign/text

fairseq-preprocess \
    --only-source \
    --trainpref "${H2S_ROOT}/train.input0.bpe" \
    --validpref "${H2S_ROOT}/val.input0.bpe" \
    --destdir "${H2S_ROOT}/input0" \
    --workers 60 \
    --srcdict dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "${H2S_ROOT}/train.label" \
    --validpref "${H2S_ROOT}/val.label" \
    --destdir "${H2S_ROOT}/label" \
    --workers 60
