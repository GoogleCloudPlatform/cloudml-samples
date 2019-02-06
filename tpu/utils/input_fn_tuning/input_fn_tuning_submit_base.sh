python trainer.py \
        --tpu=$TPU_NAME \
        --model_dir=$MODEL_DIR \
        --train_batch_size=384 \
        --max_steps=4096 \
        --save_checkpoints_steps=256 \
        {input_fn_params_str}
