import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import random
import torch
# import transformers
import torch.nn as nn
from transformers import AutoModel, BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
# from datasets import load_metric, Dataset
from sklearn.metrics import classification_report, f1_score
from pytorch_lightning import seed_everything

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

seed_everything(42)

from torch import nn
from transformers import Trainer
from sklearn.preprocessing import MinMaxScaler

drive_path = '.'

train_df = pd.read_csv(f'{drive_path}/data/df_train.csv')
val_df = pd.read_csv(f'{drive_path}/data/df_val.csv')

train_target = train_df['likes_div_views'].iloc[:100].values
val_target = val_df['likes_div_views'].iloc[:100].values

temp = np.concatenate([train_target, val_target], axis=0)
scaler = lambda x: x-temp.min() / (temp.max() - temp.min())
inv_scaler = lambda x: x * (temp.max() - temp.min()) + temp.min() 

train_text = list(train_df['text'])[:100]
val_text = list(val_df['text'])[:100]

train_df['likes_div_views'] = scaler(train_df['likes_div_views'])
val_df['likes_div_views'] = scaler(val_df['likes_div_views'])

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Tatyana/rubert-base-cased-sentiment-new")


def tokenize_function(text):
    return tokenizer(
        text,
        padding = True,
        truncation = True,
        return_tensors='pt'
    )

tokens_train = tokenizer(
        train_text,
        padding = True,
        truncation = True,
        return_tensors='pt'
    )

tokens_val = tokenizer(
        val_text,
        padding = True,
        truncation = True,
        return_tensors='pt'
    )

# class BertModel(nn.Module):
#     def __init__(self):
#         super(BertModel, self).__init__()
#         self.bert = AutoModel.from_pretrained("Tatyana/rubert-base-cased-sentiment-new")
#         self.fc = nn.Sequential(
#             nn.Linear(312,1),
#             nn.Sigmoid(),
#             )
            
#     def forward(self, *args, **kwargs):
#         output = self.bert(**kwargs)
#         emb = output.last_hidden_state[:, 0, :]
#         output['y'] = self.fc(emb)
#         return output

class Data(torch.utils.data.Dataset):
    def __init__(self, encodings, target):
        self.encodings = encodings
        self.target = target
        
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        # print(torch.tensor([self.target[idx]]))
        # print('-----------------------------')
        # print(item)
        # print('-----------------------------')
        item['labels'] = torch.tensor([self.target[idx]])
        # print('-----------------------------')
        # print(item)
        # print('-----------------------------')
        return item

    def __len__(self):
        return len(self.target)

model = AutoModel.from_pretrained("Tatyana/rubert-base-cased-sentiment-new")
train_dataset = Data(tokens_train, train_target)
test_dataset = Data(tokens_val, val_target)

from sklearn.metrics import f1_score
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds)
    return {'F1': f1}


training_args = TrainingArguments(
    output_dir = './results', #Выходной каталог
    num_train_epochs = 3, #Кол-во эпох для обучения
    per_device_train_batch_size = 1, #Размер пакета для каждого устройства во время обучения
    per_device_eval_batch_size = 1, #Размер пакета для каждого устройства во время валидации
    weight_decay =0.01, #Понижение весов
    logging_dir = './logs', #Каталог для хранения журналов
    load_best_model_at_end = True, #Загружать ли лучшую модель после обучения
    learning_rate = 1e-5, #Скорость обучения
    evaluation_strategy ='epoch', #Валидация после каждой эпохи (можно сделать после конкретного кол-ва шагов)
    logging_strategy = 'epoch', #Логирование после каждой эпохи
    save_strategy = 'epoch', #Сохранение после каждой эпохи
    save_total_limit = 1,
    seed=21)

trainer = Trainer(model=model,
                  tokenizer = tokenizer,
                  args = training_args,
                  train_dataset = train_dataset,
                  eval_dataset = test_dataset,
                  compute_metrics = compute_metrics)

trainer.train()