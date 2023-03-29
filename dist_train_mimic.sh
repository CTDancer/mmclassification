#!/bin/bash
export WANDB_API_KEY=b7b6ecceb6854bd12f58809f18264f979509d13b
./tools/dist_train.sh ./configs/resnet/resnet50_4xb32_mimic.py 4 --work-dir work_dirs/train2