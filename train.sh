#!/bin/bash

rm -rf logs/

python ./train.py \
    --workers=8 \
    --n_epochs=600 \
    --batch_size=8 \
    --n_scales=5 \
    --l_weights=0.005 \
    --div_flow=20 \
    --solver=adabound \
    --lr=2e-4 \
    --alpha=0.9 \
    --beta=0.999 \
    --final_lr=1e-4 \
    --weight_decay=1e-4 \
    --lr_step_size=60 \
    --lr_decay=0.75
