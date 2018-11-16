python trainer.py \
        --tpu=$TPU_NAME \
        --model_dir=$MODEL_DIR \
        --mode=train \
        --skip_host_call=False \
        --train_batch_size=128 \
        --train_steps=1024 \
        --num_cores=8 \
        --iterations_per_loop=256 \
        {input_fn_params_str}
