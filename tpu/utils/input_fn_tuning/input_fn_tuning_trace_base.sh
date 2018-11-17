export TPU_NAME={tpu_name}
export MODEL_DIR={model_dir}

capture_tpu_profile \
        --tpu_name=$TPU_NAME \
        --logdir=$MODEL_DIR \
        --duration_ms=15000
