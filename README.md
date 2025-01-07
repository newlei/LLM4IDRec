# LLM4IDRec
Enhancing ID-based Recommendation with Large Language Models(TOIS)

<img width="774" alt="image" src="https://github.com/newlei/LLM4IDRec/assets/16752732/7ba84429-fe23-4bc1-bf4a-7639e5bcb7ab">


Large Language Models (LLMs) have recently garnered significant attention in various domains, including recommendation systems. Recent research leverages the capabilities of LLMs to improve the performance and user modeling aspects of recommender systems. These studies primarily focus on utilizing LLMs to interpret textual data in recommendation tasks. However, it’s worth noting that in ID-based recommendations, textual data is absent, and only ID data is available. The untapped potential of LLMs for ID data within the ID-based recommendation paradigm remains relatively unexplored. To this end, we introduce a pioneering approach called "LLM for ID-based Recommendation" (LLM4IDRec). This innovative approach integrates the capabilities of LLMs while exclusively relying on ID data, thus diverging from the previous reliance on textual data. The basic idea of LLM4IDRec is that by employing LLM to augment ID data, if augmented ID data can improve recommendation performance, it demonstrates the ability of LLM to interpret ID data effectively, exploring an innovative way for the integration of LLM in ID-based recommendation. Specifically, we first define a prompt template to enhance LLM’s ability to comprehend ID data and the ID-based recommendation task. Next, during the process of generating training data using this prompt template, we develop two efficient methods to capture both the local and global structure of ID data. We feed this generated training data into the LLM and employ LoRA for fine-tuning LLM. Following the fine-tuning phase, we utilize the fine-tuned LLM to generate ID data that aligns with users’ preferences. We design two filtering strategies to eliminate invalid generated data. Thirdly, we can merge the original ID data with the generated ID data, creating augmented data. Finally, we input this augmented data into the existing ID-based recommendation models without any modifications to the recommendation model itself. We evaluate the effectiveness of our LLM4IDRec approach using three widely-used datasets. Our results demonstrate a notable improvement in recommendation performance, with our approach consistently outperforming existing methods in ID-based recommendation by solely augmenting input data.


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
### Run Recommendation model

After generating data by LLM4IDRec, you need to process and extract the corresponding interaction data, which can be achieved with the following script:
```bash
python generate_data_process.py

```


We are doing data augmentation and do not rely on specific recommendation models. The recommendation model uses open-source frameworks and code.

For BPR、SimGCL、NCL、XSimGCL、SASRec、BERT4Rec and CL4SRec, we use the open-source frameworks from the [link](https://github.com/Coder-Yu/SELFRec)
```bash
git clone https://github.com/Coder-Yu/SELFRec.git
# replace the data by augmented data, and yelp as an example
cp augmented_data_yelp.txt ./SELFRec/dataset/yelp2018/train.txt
# runing SimGCL model as example
cd SELFRec
python main.py
# Enter ‘SimGCL’ in the terminal to run SimGCL. Similarly, input various model names to run different models.
```

For P5, we use the code [link](https://github.com/agiresearch/OpenP5)
```bash
git clone https://github.com/agiresearch/OpenP5.git
# replace the data by augmented data, and yelp as an example
cp augmented_data_yelp.txt ./OpenP5/data/yelp
# Use the OpenP5 usage steps to generate data and run the model.
```


For CID+IID, we use the code [link](https://github.com/Wenyueh/LLM-RecSys-ID)
```bash
git clone https://github.com/Wenyueh/LLM-RecSys-ID.git
cp augmented_data_yelp.txt ./LLM-RecSys-ID/data/yelp
# Follow the running script from the code [link](https://github.com/Wenyueh/LLM-RecSys-ID) to generate data and execute the CID+IID.
```



### Note

We use the Llama 2-7B. Please obtain and deploy the Llama 2-7B locally from the [link](https://github.com/meta-llama/llama)

Generating training data by prompt template
```python
#!./LLM4IDRec
cd LLM4IDRec
python data_process.py
```










