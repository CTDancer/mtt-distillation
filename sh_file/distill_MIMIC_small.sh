#!/bin/bash
export WANDB_API_KEY=b7b6ecceb6854bd12f58809f18264f979509d13b
CUDA_VISIBLE_DEVICES=8 python distill.py --dataset=MIMIC_small --model=ResNet50 \
 --ipc=1 \
 --pix_init=noise \
 --num_eval=3 \
 --epoch_eval_train=1000 \
 --syn_steps=40 \
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
 --buffer_path=/home/dqwang/scratch/tongchen/mtt-distillation/buffer_MIMIC_interval=1 --data_path={path_to_dataset}
