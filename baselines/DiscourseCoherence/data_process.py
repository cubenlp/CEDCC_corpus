from cProfile import label
from copyreg import pickle
import os
from re import I
import sys
import json
from collections import defaultdict
from tqdm import tqdm
from transformers import DataProcessor,BertTokenizerFast
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import pandas as pd
import heapq
from collections import defaultdict
import pickle


class EEProcessor(DataProcessor):
    """
        从数据文件读取数据，生成训练数据dataloader，返回给模型
    """
    def __init__(self, config, tokenizer=None):
        self.train_data = self.read_data(config.train_path)
        self.dev_data = self.read_data(config.dev_path) 
        self.test_data = self.read_data(config.test_path)

        # self.test_data = self.read_test_data(config.test_path)
        
        self.tokenizer = tokenizer
        # special_tokens_dict = { 'additional_special_tokens': ["<Conn-start>", "<Conn-end>", "<single-conn-start>","<single-conn-end>"] }  # 在词典中增加特殊字符
        # self.tokenizer.add_special_tokens(special_tokens_dict)
        
        self.label2ids, self.id2labels = self._load_schema()

    def _load_schema(self):
        label2ids = {}
        id2labels = {}
       
        type_list = ["0","1","2"]
        for index,role in enumerate(type_list):
            label2ids[role] = index
            id2labels[index] = role
        return label2ids,id2labels


    def read_data(self,path):
        f = open(path,"r")
        data = json.load(f)
        return data
    
    def process_data(self,data):
        text = []
        
        label = []
        for d in data:
            # n = len(d["text"])//2
            # text_s.append("\n".join(d["text"][:n]))
            # text_e.append("\n".join(d["text"][n:]))
            # tmp = []
            # for t in d["text"]:
            #     l = t.split("。")
            #     tmp.append(l[0]+"。")
            text.append("\n".join(d["text"]))
            label.append(d["logicGrade"])
        return text,label



    def create_dataloader(self,data,batch_size,shuffle=False,max_length=512):
        tokenizer = self.tokenizer

        # data = data[:100]
        
       
        text,label = self.process_data(data)

        max_length = min(max_length,max([len(tokenizer.encode(s)) for s in text]))
        print("max sentence length: ", max_length)
        print("over 512 lengths",len([s for s in text if len(tokenizer.encode(s))>512]),len(text))

        # max_length = 512

        inputs = tokenizer(     # 得到文本的编码表示（句子前后会加入<cls>和<sep>特殊字符，并且将句子统一补充到最大句子长度
            text,
            max_length=max_length,
            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
            )

        # 4. 将得到的句子编码和BIO转为dataloader，供模型使用
        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(inputs["input_ids"]),          # 句子字符id
            # torch.LongTensor(inputs1["token_type_ids"]),     # 区分两句话
            torch.LongTensor(inputs["attention_mask"]),     # 区分是否是pad值。句子内容为1，pad为0
            # torch.LongTensor(inputs["offset_mapping"]),  
            torch.LongTensor(label),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=0,
        )
        return dataloader

