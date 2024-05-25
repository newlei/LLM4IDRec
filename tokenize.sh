CUDA_VISIBLE_DEVICES=4 python tokenize_dataset_rows.py \
    --model_checkpoint ./7b-chat/Llama-2-7b-chat \
    --input_file /kindle/train_kindle.json \
    --prompt_key q \
    --target_key a \
    --save_name train_kindle \
    --max_seq_length 2048 \
    --skip_overlength False


######################Parameter Description##############################
#--model_checkpoint ./7b-chat/Llama-2-7b-chat \ 
#you need change the path of Llama-2-7b-chat.

#--input_file /kindle/train_kindle_t8.json 
#you need change the path of dataset
