#!/bin/bash
#SBATCH --job-name="llmstep"
# #SBATCH --account=dw87
#SBATCH --comment="eleutherai"
#SBATCH --qos=dw87
#SBATCH --partition=dw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --output=llmstep_openllama_%j.out
#SBATCH --error=llmstep_openllama_%j.out
#SBATCH --time=3-00:00:00

# BYU cluster

source /home/hailey81/miniconda3/bin/activate llmstep

which python

export LD_LIBRARY_PATH=/home/hailey81/miniconda3/envs/llmstep/lib/
export PATH=/home/hailey81/cuda_install/bin:$PATH

ln -s /home/hailey81/miniconda3/envs/llmstep/bin/gcc/ ~/.local/bin/gcc
export PATH=$HOME/.local/bin:$PATH

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

BASE_DIR=$(pwd)
TRAIN_FILE=${BASE_DIR}/data/leandojo_benchmark_4/processed/proofstep-train.jsonl
VALID_FILE=${BASE_DIR}/data/leandojo_benchmark_4/processed/proofstep-val.jsonl
MODEL=openlm-research/open_llama_3b
CONFIG=${BASE_DIR}/deepspeed_config.json

OUTDIR=./model/${MODEL}

deepspeed --include localhost:0,1,2,3,4,5,6,7  ${BASE_DIR}/tune.py \
    --deepspeed ${CONFIG} \
    --model_name_or_path ${MODEL} \
    --train_data_path ${TRAIN_FILE} \
    --valid_data_path ${VALID_FILE} \
    --fp16 \
    --output_dir ${OUTDIR} \
    --max_steps 500 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 3e-5 \
    --load_best_model_at_end 1 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "linear" \
    --logging_steps 10 \
    --logging_dir "$OUTDIR" \
    --report_to="tensorboard"
