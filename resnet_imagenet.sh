#!/bin/bash
DATE=`date +%Y-%m-%d`
optimizer='sgd'

lr=0.1
ll=1e-4
ld_factor=0.1

theta=0.01
ut=1.0
wd=0.0001

# ============ ResNet-18 on ImageNet-2012 ============
model='resnet18i'
dataset='imagenet'
check_dir="./results/imagenet"

# --- AutoDrop ---
exp_name="m${optimizer^^}-$dataset-$model-lr=$lr-ld=$ld_factor-wd=$wd-theta=$theta-ll=$ll-ut=$ut-ee=20-AutoDrop-s1"
python ./codes/main.py --lr $lr --dataset $dataset --model $model --exp_name $exp_name --checkpoints_dir $check_dir --epoch 105 \
    --dropout_rate 0.0 --ld_factor $ld_factor --seed 1 --is_auto_ld --auto_theta --theta $theta --batch_size 128 --use_momentum --momentum_value 0.9 \
    --optimizer 'sgd' --wd $wd --ins_interval 5000 --lower_lr $ll --upper_theta $ut --no_nesterov --extra_epochs 20 --var_rd

# --- Baseline ---
exp_name="m${optimizer^^}-$dataset-$model-lr=$lr-ld=$ld_factor-wd=$wd-Baseline-s1"
python ./codes/main.py --lr $lr --dataset $dataset --model $model --exp_name $exp_name --checkpoints_dir $check_dir --epoch 105 \
    --dropout_rate 0.0 --ld_factor $ld_factor --seed 1 --is_preset_ld --batch_size 128 --use_momentum --momentum_value 0.9 --no_nesterov \
    --optimizer 'sgd' --wd $wd --ins_interval 5000
