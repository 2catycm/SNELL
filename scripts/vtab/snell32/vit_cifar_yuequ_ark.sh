#!/usr/bin/env bash         
# hyperparameters
export DATASET=cifar
export SEED=0
export batch_size=64
export LR=0.001
export WEIGHT_DECAY=0.01

# export CUDA_VISIBLE_DEVICES=1
# export KERNEL_NAME=linear\*

# export CUDA_VISIBLE_DEVICES=0
# export KERNEL_NAME=tanh

# export CUDA_VISIBLE_DEVICES=2
# export KERNEL_NAME=swish

# export CUDA_VISIBLE_DEVICES=3
# export KERNEL_NAME=poly_comb

export CUDA_VISIBLE_DEVICES=1
export KERNEL_NAME=rbf



export tuning_model=yuequ_ark-$KERNEL_NAME
export low_rank_dim=32
export init_thres=0.9

export exp_name=vtab_vit_supervised_${LR}_${init_thres}_${WEIGHT_DECAY}_${low_rank_dim}_${batch_size}_200

ca yuequ

python train.py --data-path=./data/vtab-1k/${DATASET} --init_thres=${init_thres} \
 --data-set=${DATASET} --model_name=vit_base_patch16_224_in21k_snell --resume=checkpoints/ViT-B_16.npz \
 --output_dir=./saves_vtab/${tuning_model}/${DATASET}/${exp_name} \
 --batch-size=${batch_size} --lr=${LR} --epochs=200 --weight-decay=${WEIGHT_DECAY} --no_aug --mixup=0 --cutmix=0 --direct_resize \
 --smoothing=0 --launcher="none" --seed=${SEED} --val_interval=10  --opt=adamw --low_rank_dim=${low_rank_dim} \
 --exp_name=${exp_name} --seed=0 \
 --test --block=BlockSNELLParallel  --tuning_model=${tuning_model} --freeze_stage