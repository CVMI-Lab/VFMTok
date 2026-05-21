# !/bin/bash
export NODE_COUNT=$1
export NODE_RANK=$2
export PROC_PER_NODE=8
export MASTER_ADDR=100.104.220.26
export MASTER_PORT=22333
model_type='GPT-B'
scripts/autoregressive/torchrun.sh train_c2i.py --gpt-type c2i --image-size 336 --gpt-model ${model_type} --downsample-size 16 --num-workers 4   \
    --anno-file imagenet/lmdb/train_lmdb --global-batch-size 512 --ckpt-every 10000 --ema --log-every 1 --results-dir output \
    --vq-model VQ-16 --vq-ckpt DINOv2/tokenizer/vfmtok-tokenizer.pt --latent-size 16 --mixed-precision bf16     \
    --epochs 300
