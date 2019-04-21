#!/bin/bash

rm -rf logs/

python ./train.py \
    --workers=8 \
    --n_epochs=160 \
    --batch_size=32 \
    --n_scales=5 \
    --l_weights=0.005 \
    --div_flow=0.05 \
    --solver=adabound \
    --lr=5e-4 \
    --alpha=0.9 \
    --beta=0.999 \
    --final_lr=5e-3 \
    --weight_decay=5e-4 \
    --lr_step_size=40 \
    --lr_decay=0.25
