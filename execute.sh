#!/bin/bash

# Run: ./run_squad_basic.sh <outfile> to run process in background
# and redirect output to a file
# Otherwise, ./run_squad_basic.sh will run the program as usual
export SQUAD_DIR=/ml/jif24/squad

RUN_IN_BACKGROUND=${1:-"FALSE"}
if [ "$RUN_IN_BACKGROUND" != "FALSE" ]
then
    nohup python run_squad_basic.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --do_eval \
        --version_2_with_negative \
        --train_file $SQUAD_DIR/train-v2.0.json \
        --predict_file $SQUAD_DIR/dev-v2.0.json \
        --learning_rate 3e-5 \
        --num_train_epochs 4 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --output_dir ./wwm_cased_finetuned_squad/ \
        --per_gpu_eval_batch_size=2  \
        --per_gpu_train_batch_size=2   \
        --save_steps 5000 \
        --logging_steps 100 > $1 2>&1 &
else
    python run_squad_basic.py \
        --model_type bert \
        --model_name_or_path wwm_cased_finetuned_squad/checkpoint-20000/ \
        --do_eval \
        --version_2_with_negative \
        --train_file $SQUAD_DIR/train-v2.0.json \
        --predict_file $SQUAD_DIR/dev-v2.0.json \
        --learning_rate 3e-5 \
        --num_train_epochs 4 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --output_dir ./wwm_cased_finetuned_squad/ \
        --per_gpu_eval_batch_size=1  \
        --per_gpu_train_batch_size=1   \
        --save_steps 5000 \
        --logging_steps 10
fi
