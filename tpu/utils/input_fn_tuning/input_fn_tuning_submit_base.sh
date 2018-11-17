# export TPU_NAME={tpu_name}
# export MODEL_DIR={model_dir}

python trainer.py \
        --tpu=$TPU_NAME \
        --model_dir=$MODEL_DIR \
        --train_batch_size=128 \
        --max_steps=1024 \
        --save_checkpoints_steps=256 \
        {input_fn_params_str}
