# coding=utf-8
from typing import DefaultDict, Sequence
from torch._C import device, set_flush_denormal
from data_process import EEProcessor
import pytorch_lightning as pl
# from sklearn import model_selection
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
import csv
from conlleval import evaluate,RE_evaluate
from transformers import (
    BertTokenizerFast,
    BertTokenizer,
    BertForTokenClassification,
    BertModel
)
from transformers import AutoTokenizer, AutoModelForMaskedLM,BertModel,AutoModelForSequenceClassification

from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.nn import CrossEntropyLoss

import re
import json
import pandas as pd
from sklearn.metrics import classification_report

class EEModel(pl.LightningModule):
    def __init__(self, config):
        # 1. Init parameters
        super(EEModel, self).__init__()
        
        self.config=config

        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_path)  # 
        # self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_path)
        special_tokens_dict = { 'additional_special_tokens': ["\n"] }  # 在词典中增加特殊字符

        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.processor = EEProcessor(config, self.tokenizer)

        self.labels = len(self.processor.label2ids)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.pretrained_path, num_labels=self.labels)

        # self.model = BertModel.from_pretrained(
        #     config.pretrained_path, num_labels=self.labels)
        self.model.resize_token_embeddings(len(self.tokenizer))


        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size)
        self.classifier = nn.Linear(self.model.config.hidden_size, self.labels)
        self.loss_fct = CrossEntropyLoss()
        self.batch_size = config.batch_size
        self.optimizer = config.optimizer
        self.lr = config.lr


    def prepare_data(self):
        train_data = self.processor.train_data
        dev_data = self.processor.dev_data
        # dev_data = self.processor.train_data
        if self.config.train_num>0:
            train_data=train_data[:self.config.train_num]
        if self.config.dev_num>0:
            dev_data=dev_data[:self.config.dev_num]

        print("train_length:", len(train_data))
        print("valid_length:", len(dev_data))

        self.train_loader = self.processor.create_dataloader(
            train_data, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = self.processor.create_dataloader(
            dev_data, batch_size=self.batch_size, shuffle=False)

    def forward(self, input_ids, attention_mask):
        feats = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]  # [bs,len,numlabels]
        # # pooler_output = self.model(
        # #     input_ids=input_ids,
        # #     attention_mask=attention_mask
        # # ).pooler_output
        # embedding1=self.model(input_ids=input_ids,attention_mask=attention_mask).pooler_output
        

        # rep = self.layer_norm(embedding1)
        # rep = self.dropout(rep)
        # logits = self.classifier(rep)
        # return logits

        # # feats = self.model(
        # #     input_ids=input_ids,
        # #     attention_mask=attention_mask,
        # # )[0]  # [bs,len,numlabels]

        return feats


    def training_step(self, batch, batch_idx):
        input_ids1, attention_mask1, label = batch
        logits = self.forward(input_ids1, attention_mask1)
        loss = self.loss_fct(logits,label)
        # loss = self.loss_fct(logits.view(-1, self.labels), label.view(-1))
        self.log('train_loss', loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids1, attention_mask1, label = batch
        logits = self.forward(input_ids1, attention_mask1)
        # loss = self.loss_fct(logits.view(-1, self.labels), label.view(-1))
        loss = self.loss_fct(logits,label)
        pred = logits.argmax(dim=-1)

        gold = label
        pre = torch.tensor(pred).cuda()

        return loss.cpu(),gold.cpu(),pre.cpu()

    def validation_epoch_end(self, outputs):
        val_loss,gold,pre = zip(*outputs)

        val_loss = torch.stack(val_loss).mean()
        gold = torch.cat(gold)
        pre = torch.cat(pre)


        # # print("")
        # def to_label(label,con):
        #     if label==0 :return "O"
        #     return "B-"+self.processor.ids2conn[con.item()]

        true_seqs = [self.processor.id2labels[int(g)] for g in gold]
        pred_seqs = [self.processor.id2labels[int(g)] for g in pre]
        
        print(classification_report(
                y_pred=pre, y_true=gold, digits=4, 
                labels = [x for x in range(0, len(self.processor.label2ids.keys()))],
                target_names = list(self.processor.label2ids.keys())
                ))

       

        print("true_seqs",len(true_seqs),true_seqs[:5])
        print("pred_seqs",len(pred_seqs),pred_seqs[:5])

        print('\n')
        prec, rec, f1 = RE_evaluate(true_seqs, pred_seqs)

        self.log('val_loss', val_loss)
        self.log('val_pre', torch.tensor(prec))
        self.log('val_rec', rec)
        self.log('val_f1', torch.tensor(f1))

    def configure_optimizers(self):
        # if self.use_crf:
        #     crf_params_ids = list(map(id, self.crf.parameters()))
        #     base_params = filter(lambda p: id(p) not in crf_params_ids, [
        #                          p for p in self.parameters() if p.requires_grad])

        #     arg_list = [{'params': base_params}, {'params': self.crf.parameters(), 'lr': self.crf_lr}]
        # else:
        #     # label_embed_and_attention_params = list(map(id, self.label_embedding.parameters())) + list(map(id, self.self_attention.parameters()))
        #     # arg_list = [{'params': list(self.label_embedding.parameters()) + list(self.self_attention.parameters()), 'lr': self.lr}]
        #     arg_list = [p for p in self.parameters() if p.requires_grad]
        arg_list = [p for p in self.parameters() if p.requires_grad]

        print("Num parameters:", len(arg_list))
        if self.optimizer == 'Adam':
            return torch.optim.Adam(arg_list, lr=self.lr, eps=1e-8)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(arg_list, lr=self.lr, momentum=0.9)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

class EEPredictor:
    def __init__(self, checkpoint_path, config):
        self.model = EEModel.load_from_checkpoint(checkpoint_path, config=config)
        self.tokenizer = self.model.tokenizer
        self.batch_size = config.batch_size
        self.test_data = self.model.processor.test_data
        # self.en_zh_mapping = {
        #     "Causation":"因果关系",
        #     "Alternative":"选择关系",
        #     "Contrast":"对比关系",
        #     "Conjunction":"并列关系",
        #     "Conditional":"条件关系",
        #     "Progression":"递进关系",
        #     "Purpose":"目的关系",
        #     "Expansion":"解说关系",
        #     "Temporal":"时序关系"
        # }
        print('load checkpoint:', checkpoint_path)

    def predict(self):
        f = open("/home/hongyi/EMNLP_2023/all_datas_final.json","r")
        datas = json.load(f)
        id2grade = {}
        
        for RelationType in [["显式"],["隐式"],["显式","隐式"]]:
            test_data = [d for d in self.test_data if d["RelationType"]in RelationType and d[""]]
            self.dataloader = self.model.processor.create_dataloader(
            test_data,batch_size=self.batch_size,shuffle=False)

            print("The TEST num is:", len(test_data))
        
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            # device = torch.device("cpu")

            self.model.to(device)
            self.model.eval()
            print(RelationType)

            print(len(test_data),len(self.dataloader))

            pred_list = []
            label_list = []
            for batch in tqdm.tqdm(self.dataloader):
                for i in range(len(batch)):
                    batch[i] = batch[i].to(device)

                input_ids1, attention_mask1, label = batch
                logits = self.model(input_ids1, attention_mask1)
                preds = logits.argmax(dim=-1)
                # for pred in list(preds):
                #     # if pred != 0:
                #     zh_pred = self.model.processor.id2labels[int(pred)]
                #     out_list.append(zh_pred)
                pred_list.extend(preds.cpu())
                label_list.extend(label.cpu())

            # print(len(out_list))

            # print(out_list[:5])
            print(classification_report(
                            y_pred=pred_list, y_true=label_list, digits=4, 
                            labels = [x for x in range(0, len(self.model.processor.label2ids.keys()))],
                            target_names = list(self.model.processor.label2ids.keys())
                            ))
        return pred_list

    def generate_result(self,outfile):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # device = torch.device("cpu")

        self.model.to(device)
        self.model.eval()

        print(len(self.test_data),len(self.dataloader))

        out_list = []
        for batch in tqdm.tqdm(self.dataloader):
            for i in range(len(batch)):
                batch[i] = batch[i].to(device)
            input_ids1, attention_mask1, conn_input_ids, conn_attention_mask, label = batch
            logits = self.forward(input_ids1, attention_mask1, conn_input_ids, conn_attention_mask)
            preds = logits.argmax(dim=-1)
            for pred in list(preds):
                # if pred != 0:
                out_list.append(self.model.processor.id2labels[int(pred)])
        print(len(out_list))

        print(out_list[:5])
        self.model.processor.test_data["pred"] = out_list
        self.model.processor.test_data.to_csv(outfile,index=0)






