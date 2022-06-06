# The name of experiment
export CUDA_VISIBLE_DEVICES=0,1,2,3
name=scst_exp
dataset=$2
split=$3

output=snap/$dataset/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/reg.py \
        --distributed --multiGPU \
        --train train \
        --valid val \
        --test test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 1e-6 \
        --epochs 1 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output \
        --load snap/$dataset/REG/BEST \
        --num_beams 5 \
        --batch_size 32 \
        --valid_batch_size 100 \
        --dataset $dataset\
        --dataset_split $split\
        --experiment_name $name\
        --rl_training\
        --debug\
        # --use_mmi \