import random
import json
import pdb
import numpy as np
from tqdm import tqdm

item_num = 38048+1
user_num = 31668+1
num_neg = 50
num_pos = 5


train_llm_path = './augmented_data_yelp.txt'
print(train_llm_path)

f_train_llm = open(train_llm_path,"w")  

train_data_path = './data/yelp/rain.txt'
f_train = open(train_data_path,"r") 
lines_t = f_train.readlines() 
f_train.close()
lines_train = dict()
lines_train_pos = dict()
for line in lines_t:
    list_one = line.strip().split(' ')
    user_id = list_one[0]
    positive_list= list_one[1] 
    if user_id in lines_train:
        lines_train[user_id].add(int(positive_list))#= positive_list
    else:
        lines_train[user_id]=set()
        lines_train[user_id].add(int(positive_list))


lines_train_add = dict()
path_llm = '../eval/**_predictions.json'#LLM4IDRec ouput results path
with open(path_llm, "r") as f:
    for line in tqdm(f.readlines()):
        # pdb.set_trace()
        example = json.loads(line.strip())  
        prompt = example["prompt"]
        prediction = example["prediction"]
        s_p = prompt.find('user(u')+6
        e_p = prompt.find(')')
        user_id = prompt[s_p:e_p] 

        item_lists = prediction.split('i') 
        str_neg = str(user_id)+' '
        add_items = set() 

        for one_item in item_lists:#[:-1]:#[1:-1]:
            try:
                idx_split=one_item.find(",")
                if idx_split>=0: 
                    item_id = int(one_item[:idx_split])
                else:
                    item_id = int(one_item) 
                if item_id<item_num:
                    add_items.add(item_id) 
            except:
                continue  
        if user_id in lines_train_add:
            lines_train_add[user_id].update(add_items)  
        else:
            lines_train_add[user_id] = add_items  




for id in lines_train:
    str_id = str(id)
    
    if id in lines_train_add:
        lines_train[id].update(lines_train_add[id])   
    for i in lines_train[id]:
        str_id+=' '+str(i)
    f_train_llm.write(str_id)
    f_train_llm.write("\n")
    f_train_llm.flush()  

f_train_llm.close()

exit()
