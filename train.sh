#!/bin/bash

rm -rf logs/

python ./train.py \
    --workers=8 \
    --n_epochs=300 \
    --batch_size=8 \
    --lr=1e-4 \
    --n_scales=5 \
    --l_weights=0.32 \
    --div_flow=0.05 \
    --alpha=0.9 \
    --beta=0.999 \
    --weight_decay=4e-4 \
    --lr_step_size=50 \
    --lr_decay=0.1
