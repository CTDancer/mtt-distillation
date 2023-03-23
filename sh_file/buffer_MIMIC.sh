#!/bin/bash
export WANDB_API_KEY=b7b6ecceb6854bd12f58809f18264f979509d13b
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python buffer.py --dataset=MIMIC --model=ResNet18 --train_epochs=50 --num_experts=10 --lr_teacher=0.001 --batch_train=16 --batch_real=16 --decay --mom=0.9 --l2=0.01 --save_interval=10 --dsa=True --buffer_path=buffer_MIMIC --data_path={path_to_dataset}