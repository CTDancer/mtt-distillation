#!/bin/bash
export WANDB_API_KEY=b7b6ecceb6854bd12f58809f18264f979509d13b
CUDA_VISIBLE_DEVICES=5 python buffer2.py --dataset=MIMIC --model=ResNet18 --train_epochs=50 --num_experts=10 --lr_teacher=0.00001 --batch_train=512 --batch_real=512 --decay --mom=0.9 --l2=0.0001 --save_interval=10 --buffer_path=buffer_MIMIC --data_path={path_to_dataset}
