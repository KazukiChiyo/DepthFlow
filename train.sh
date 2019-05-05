#!/bin/bash

rm -rf logs/

python ./train.py \
    --workers=12 \
    --n_epochs=500 \
    --batch_size=8 \
    --n_scales=5 \
    --l_weights=0.005 \
    --div_flow=20 \
    --solver=adam \
    --lr=1e-4 \
    --alpha=0.9 \
    --beta=0.999 \
    --final_lr=1e-4 \
    --weight_decay=4e-4 \
    --bias_decay=0 \
    --milestones 100 150 200 \
    --lr_decay=0.5 \
    --depth=True
