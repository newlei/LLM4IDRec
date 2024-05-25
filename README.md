# LLM4IDRec
Enhancing ID-based Recommendation with Large Language Models


We provide code for LLM4IDRec model.


## Prerequisites

- Llama 2 and transformers
- PyTorch
- Python 3.5
- CPU or NVIDIA GPU + CUDA CuDNN


## Getting Started

### Installation

- Clone this repo:

```bash
git clone https://github.com/newlei/LLM4IDRec.git
cd LLM4IDRec
```


### Fine-turning and Inference


```bash
#!./LLM4IDRec
cd LLM4IDRec
#Data preprocessing: we utilized the Llama2 tokenizer to process the data.
bash tokenize.sh
#Fine-turning LLM: we employ LoRA as an efficient means to fine-turning pretrained LLM.
bash lora_tuning.sh
#Inference for data generation
bash predict.sh
```

### Note

We use the Llama 2-7B. Please obtain and deploy the Llama 2-7B locally from the [link](https://github.com/meta-llama/llama)










