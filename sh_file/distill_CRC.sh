#!/bin/bash
export WANDB_API_KEY=b7b6ecceb6854bd12f58809f18264f979509d13b
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python distill.py --dataset=CRC1 --model=ResNet18 \
 --ipc=5 \
 --pix_init=noise \
 --num_eval=3 \
 --epoch_eval_train=1000 \
 --syn_steps=50 \
 --expert_epochs=2 \
 --batch_train=4 \
 --batch_real=4 \
 --max_start_epoch=1 \
 --lr_init=1e-5 \
 --lr_img=10000 \
 --lr_lr=1e-09 \
 --lr_teacher=0.00001 \
 --img_wd=0 \
 --lr_wd=0 \
 --img_mom=0.5 \
 --lr_mom=0.5 \
 --buffer_path=buffer_long --data_path={path_to_dataset}
