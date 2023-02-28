#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python buffer.py --dataset=CRC_small --model=ResNet18 --train_epochs=50 --num_experts=10 --lr_teacher=0.001 --batch_train=16 --batch_real=16 --decay --mom=0.9 --l2=0.01 --save_interval=1 --buffer_path={path_to_buffer_storage} --data_path={path_to_dataset}