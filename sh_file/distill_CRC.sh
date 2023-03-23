#!/bin/bash
CUDA_VISIBLE_DEVICES=7,6,5,4,3 python distill.py --dataset=CRC1 --model=ResNet18 --ipc=5 --pix_init=noise --num_eval=3 --epoch_eval_train=5000 --syn_steps=60 --batch_train=4 --batch_real=4 --expert_epochs=3 --max_start_epoch=5 --lr_init=1e-5 --lr_img=10000 --lr_lr=1e-09 --lr_teacher=0.00001 --buffer_path=buffer_long --data_path={path_to_dataset}
