# Download encoder.json and vocab.bpe
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'

H2S_ROOT=../../../data/How2Sign/text
for SPLIT in train val test; do
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs ${H2S_ROOT}/${SPLIT}.input0 \
        --outputs ${H2S_ROOT}/${SPLIT}.input0.bpe \
        --workers 60 \
        --keep-empty
done
