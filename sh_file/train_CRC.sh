#!bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python buffer.py --dataset=CRC_small --model=ConvNet --train_epochs=50 --num_experts=100 --buffer_path={path_to_buffer_storage} --data_path={path_to_dataset}