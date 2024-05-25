
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
import transformers
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
import os
from pprint import pprint as print
import pdb
from shutil import copyfile   
import sys

model_path = "./7b-chat/Llama-2-7b-chat"   #"../Llama-2-7b" # "./7b-hf" #  "../Llama-2-7b"   # "../Llama-2-7b" #"THUDM/chatglm-6b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


@dataclass
class FinetuneArguments:
    tokenized_dataset: str = field(default=" ") # tokenized之后的数据集文件夹
    model_path: str = field(default=" ")
    lora_rank: int = field(default=8) 


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


tokenizer.pad_token = tokenizer.unk_token
def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"] # prompt length
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)  
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    #tokenizer.batch_decode(input_ids)
    #tokenizer.batch_decode(labels)

    return {
        "input_ids": input_ids,
        "labels": labels,
    }
# 这里的 collator 主要参考了 https://github.com/mymusise/ChatGLM-Tuning/blob/master/finetune.py 中的写法
# 将 prompt 的部分的label也设置为了 -100，从而在训练时不纳入loss的计算
# 对比之下，我在 baichaun_lora_tuning.py 中，是直接使用 DataCollatorForLanguageModeling，prompt 也纳入了计算。
# 这两种方式孰优孰劣尚不得而知，欢迎在issue或discussion中讨论。

class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        # 因为交给Trainer的model实际上是PeftModel类型，所以这里的 save_pretrained 会直接使用PeftModel的保存方法
        # 从而只保存 LoRA weights
        self.model.save_pretrained(output_dir)
        # from transformers.trainer import TRAINING_ARGS_NAME
        # os.makedirs(output_dir, exist_ok=True)
        # torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        # saved_params = {
        #     k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        # }
        # torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))
        


def main():
    writer = SummaryWriter()
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    
        
    path_save_base= training_args.output_dir
    if (os.path.exists(path_save_base)):
        print('has results save path')
    else:   
        os.makedirs(path_save_base)
    result_file=path_save_base+'/results.txt'#open(path_save_base+'/results.txt','a')#('./log/results_gcmc.txt','w+') 
    copyfile('./lora_tuning.py', path_save_base+'/lora_tuning.py') 
    copyfile('./lora_tuning.sh', path_save_base+'/lora_tuning.sh') 

    class Logger(object):
        def __init__(self, filename='default.log', stream=sys.stdout):
            self.terminal = stream
            self.log = open(filename, 'a')

        def write(self, message):
            self.terminal.write(message)
            if message.find('%')==-1:
                self.log.write(message)
                self.log.flush()

        def flush(self):
            pass
    sys.stdout = Logger(filename = result_file,stream=sys.stdout)


    # load dataset
    dataset = datasets.load_from_disk('data/tokenized_data/'+finetune_args.tokenized_dataset)
    dataset = dataset.shuffle(seed=2023) #我加上了随机数，不然数据都是按照顺序去读，就容易产生连续的值。
    # dataset = dataset.select(range(10000))
    print(f"\n{len(dataset)=}\n")
    
    # init model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, load_in_8bit=False, trust_remote_code=True, 
        device_map="auto" # 模型不同层会被自动分配到不同GPU上进行计算
        # device_map={'':torch.cuda.current_device()}
    )
    print(model.hf_device_map)
    
    """
    .gradient_checkpointing_enable()
    .enable_input_require_grads()
    .is_parallelizable
    这三个都是 transformers 模型的函数/参数（见 transformers/modeling_utils.py 文件）
    """
    model.gradient_checkpointing_enable() 
    # note: use gradient checkpointing to save memory at the expense of slower backward pass.
    model.enable_input_require_grads()
    # note: Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping the model weights fixed. 
    # See https://github.com/huggingface/transformers/blob/ee88ae59940fd4b2c8fc119373143d7a1175c651/src/transformers/modeling_utils.py#L1190
    # model.is_parallelizable = True
    # note: A flag indicating whether this model supports model parallelization.
    # model.model_parallel = True # 在transformers和ChatGLM的modeling_chatglm.py中好像没看到有这个参数
    # pdb.set_trace() 
    model.lm_head = CastOutputToFloat(model.lm_head)
    # model.config.use_cache = (
    #     False  # silence the warnings. Please re-enable for inference!
    # )

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        lora_alpha= 32,
        lora_dropout= 0.05,
        target_modules = ["q_proj","v_proj"],
    )
    
    
    model = get_peft_model(model, peft_config)


    # start train
    model.save_pretrained(training_args.output_dir) # 
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        # data_collator=data_collator,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False), 
    )
    trainer.train()
    writer.close()
    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()