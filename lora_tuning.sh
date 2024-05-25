CUDA_VISIBLE_DEVICES=4 python lora_tuning.py \
    --tokenized_dataset train_kindle \
    --lora_rank 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_steps 400 \
    --save_steps 50 \
    --save_total_limit 2 \
    --learning_rate 1e-3 \
    --fp16 True \
    --remove_unused_columns false \
    --logging_steps 10 \
    --output_dir weights/train_kindle
