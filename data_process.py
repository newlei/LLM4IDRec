import random
import json
import pdb
import numpy as np
from random import shuffle,randint,choice,sample

def file2data(filename): 
    data_dict = dict()
    f = open(filename,"r") 
    lines = f.readlines() 
    f.close()
    for line in lines:
        list_one = line.strip().split(' ') 
        user_id = list_one[0]
        if user_id not in data_dict:
            data_dict[user_id]=[]
        data_dict[user_id].append(list_one[1])
 
    return  data_dict


def generate_data(train_data_path,output_json,test_flag=False):
    f_output = open(output_json, 'w')

    # f = open(train_data_path,"r") 
    # lines = f.readlines() 
    # f.close()
    data_train = file2data(train_data_path)

    Prompt_json = []
    for user_i in data_train:
        user_id = 'u'+user_i
        positive_list= data_train[user_i]
        len_positive_list = len(positive_list) 
        if len_positive_list<3:
            continue 

        rand_sel = np.random.randint(len_positive_list-2, size=20)

        for item_j in rand_sel: #range(start_id,len_positive_list):
            item_i = item_j+1
            random.shuffle(positive_list) 
            preference_str =''
            #train_yelp_qa_pn7,下面3行是针对pn7做的修改，是为了让模型输入更少。方便batch size更大。 

            for i in range(item_i):
                preference_str +='i'+positive_list[i]+','
            preference_str = preference_str[:-1]

            pos_str = '' 
            for i_pos in range(item_i,len_positive_list):
                pos_str +='i'+positive_list[i_pos]+','
            pos_str = pos_str[:-1]
            
            str_out = {"q": f"Given the user({user_id})'s clicked list items:{preference_str}, predict what is the list items to recommend to the user({user_id}). Please only answer the item IDs.","a": f"{target_preference_str}"}
            json.dump(str_out, f_output)
            f_output.write("\n")
            f_output.flush() 


#train.txt is the train data of amazon-kindle
generate_data('./train.txt','./train_kindle.json')  
print('train end')

#You can use the code to generate test_kindle.json.txt
