# NOTICE:
# mode: baseline, input-first, textual-label, mentioned-only, lines-output, jsonfy-output


CUDA_VISIBLE_DEVICES=4 python predict.py\
    --llm_ckp  ./7b-chat/Llama-2-7b-chat\
    --lora_path weights/train_kindle \
    --data_path ./data/kindle/test_kindle.json \
    --prompt_key q \
    --target_key a \
    --batch_size 16\
    --id test_t1
 