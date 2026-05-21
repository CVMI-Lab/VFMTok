# !/bin/bash
export NODE_COUNT=1
export NODE_RANK=0
export PROC_PER_NODE=8
export MASTER_PORT=22333
scripts/autoregressive/torchrun.sh vqgan_test.py --vq-model VQ-16 --image-size 336 --output_dir recons --batch-size $1   \
        --z-channels 512 --vq-ckpt pretrained_models/vfmtok-tokenizer.pt --codebook-slots-embed-dim 12
