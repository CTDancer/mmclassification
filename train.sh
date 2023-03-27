#!/bin/bash
export WANDB_API_KEY=b7b6ecceb6854bd12f58809f18264f979509d13b
export CUDA_VISIBLE_DEVICES=5,6 
python ./tools/train.py ./configs/resnet/resnet18_4x512_mimic.py --work-dir work_dirs/train1 