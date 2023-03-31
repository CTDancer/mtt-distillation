#!/bin/bash
export WANDB_API_KEY=b7b6ecceb6854bd12f58809f18264f979509d13b
CUDA_VISIBLE_DEVICES=6 python buffer.py \
--dataset=MIMIC \
--model=ResNet50 \
--train_epochs=50 \
--num_experts=10 \
--lr_teacher=0.0005 \
--batch_train=32 \
--decay \
--mom=0.9 \
--l2=1e-3 \
--save_interval=10 \
--buffer_path=buffer_MIMIC_4 --data_path={path_to_dataset}
