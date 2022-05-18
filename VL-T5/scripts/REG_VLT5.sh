# The name of experiment
name=REG_test
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
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output \
        --load snap/pretrain/VLT5/Epoch30 \
        --num_beams 1 \
        --batch_size 80 \
        --valid_batch_size 100 \
        --dataset $dataset\
        --dataset_split $split\
        --experiment_name prefix_exp\
        # --use_mmi \