#!/bin/bash

# Run: ./run_squad_basic.sh <outfile> to run process in background
# and redirect output to a file
# Otherwise, ./run_squad_basic.sh will run the program as usual
export SQUAD_DIR=/ml/jif24/squad

CHECKPOINT_DIR=$1
OUTPUT_FILE=${2:-"FALSE"}
if [ "$OUTPUT_FILE" != "FALSE" ]
then
    nohup python run_brainqa.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --version_2_with_negative \
        --train_file $SQUAD_DIR/train-v2.0.json \
        --predict_file $SQUAD_DIR/dev-v2.0.json \
        --learning_rate 3e-5 \
        --num_train_epochs 4 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --output_dir ./$CHECKPOINT_DIR \
        --per_gpu_eval_batch_size=2  \
        --per_gpu_train_batch_size=2   \
        --save_steps 5000 \
        --logging_steps 100 > $OUTPUT_FILE 2>&1 &
        --do_interpolate
else
    python run_brainqa.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --version_2_with_negative \
        --train_file $SQUAD_DIR/train-v2.0.json \
        --predict_file $SQUAD_DIR/dev-v2.0.json \
        --learning_rate 3e-5 \
        --num_train_epochs 4 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --output_dir ./$CHECKPOINT_DIR \
        --per_gpu_eval_batch_size=2  \
        --per_gpu_train_batch_size=2   \
        --save_steps 5000 \
        --logging_steps 10 \
        --do_interpolate
fi
