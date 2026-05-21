# !/bin/bash
export NODE_COUNT=1
export NODE_RANK=0
export PROC_PER_NODE=8
model_type='GPT-3B'
scripts/autoregressive/torchrun.sh test_net.py --vq-ckpt DINOv2/tokenizer/vfmtok-tokenizer.pt            \
    --gpt-ckpt DINOv2/${model_type}/${model_type}-$1e.pt --compile --gpt-model ${model_type} --image-size 336 \
    --sample-dir samples --image-size-eval 256 --cfg-scale $2 --precision bf16 --per-proc-batch-size $3   \
    --codebook-slots-embed-dim 12 --latent-size 16