# 看cuda 是否可用
from multiprocessing import freeze_support

import torch

# print(torch.cuda.is_available())

device = torch.device("cuda")

# 一堆 import

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# set a seed value
torch.manual_seed(555)

from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score

import transformers
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaModel
from transformers import AdamW

from transformers import XLMRobertaTokenizer, XLMRobertaModel
tokenizer = XLMRobertaTokenizer.from_pretrained("hfl/cino-base-v2")
model = XLMRobertaModel.from_pretrained("hfl/cino-base-v2")
rTokenizer = XLMRobertaTokenizer.from_pretrained('hfl/cino-base-v2')
# print(rTokenizer.vocab_size)
#
# # data
# ## import
# import pandas as pd
# import numpy as np
# from sklearn.metrics import f1_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# import gc
#
# urduRaw = pd.read_csv("D:\\all_project\\jupyter_workplace\\paper_reproduction\\XLM-R\\dataset\\urdu.tsv", sep='\t')
# spanishRaw = pd.read_csv("D:\\all_project\\jupyter_workplace\\paper_reproduction\\XLM-R\\dataset\\spanish.csv")
# marathiRaw = pd.read_csv("D:\\all_project\\jupyter_workplace\\paper_reproduction\\XLM-R\\dataset\\marathi.csv")
# chineseRaw = pd.read_csv("D:\\all_project\\jupyter_workplace\\paper_reproduction\\XLM-R\\dataset\\chinese.csv")
# englishRaw = pd.read_csv("D:\\all_project\\jupyter_workplace\\paper_reproduction\\XLM-R\\dataset\\english.tsv",
#                          sep='\t')
#
# ## data cleaning
# ### spanish
# spanish = spanishRaw[["text", "airline_sentiment"]].copy(deep=True).sample(frac=1).reset_index(drop=True)
# spanish.rename(columns={'airline_sentiment': 'label'}, inplace=True)
# spanish = spanish[spanish['label'] != "neutral"]
# spanish.loc[(spanish.label == "negative"), "label"] = 0
# spanish.loc[(spanish.label == 'positive'), "label"] = 1
# spanish['language'] = 'spanish'
# # print(spanish['label'].value_counts())
#
# val_sizeS = int(spanish.shape[0] * 0.15)
# test_sizeS = int(spanish.shape[0] * 0.15)
# test_spanish = spanish[:test_sizeS].reset_index(drop=True)
# val_spanish = spanish[test_sizeS:test_sizeS + val_sizeS].reset_index(drop=True)
# train_spanish = spanish[test_sizeS + val_sizeS:].reset_index(drop=True)
#
# ### Urdu
#
# urdu = urduRaw.copy(deep=True).sample(frac=1).reset_index(drop=True)
# urdu.rename(columns={'Tweet': 'text', 'Class': 'label'}, inplace=True)
# urdu['language'] = 'urdu'
# urdu = urdu[urdu['label'].notna()]
# urdu = urdu[urdu['label'] != 'O']
# urdu.loc[(urdu.label == "N"), "label"] = 0
# urdu.loc[(urdu.label == 'P'), "label"] = 1
#
# # print(urdu['label'].value_counts())
#
# val_sizeU = int(urdu.shape[0] * 0.15)
# test_sizeU = int(urdu.shape[0] * 0.15)
# test_urdu = urdu[:test_sizeU].reset_index(drop=True)
# val_urdu = urdu[test_sizeU:test_sizeU + val_sizeU].reset_index(drop=True)
# train_urdu = urdu[test_sizeU + val_sizeU:].reset_index(drop=True)
#
# ### Marathi
# marathi = marathiRaw.copy(deep=True).sample(frac=1).reset_index(drop=True)
# marathi.rename(columns={'tweet': 'text'}, inplace=True)
# marathi = marathi[marathi['label'] != 0]
# marathi.loc[(marathi.label == -1), "label"] = 0
# marathi['language'] = 'marathi'
# # print(marathi['label'].value_counts())
#
# val_sizeM = int(marathi.shape[0] * 0.15)
# test_sizeM = int(marathi.shape[0] * 0.15)
# test_marathi = marathi[:test_sizeM].reset_index(drop=True)
# val_marathi = marathi[test_sizeM:test_sizeM + val_sizeM].reset_index(drop=True)
# train_marathi = marathi[test_sizeM + val_sizeM:].reset_index(drop=True)
#
# ### Chinese
# chinese = chineseRaw[["review", "label"]].copy(deep=True).sample(frac=1).reset_index(drop=True)
# chinese.rename(columns={'review': 'text'}, inplace=True)
# chinese['language'] = 'chinese'
# chinese = chinese.sample(9000).reset_index(drop=True).copy()
# # print(chinese['label'].value_counts())
#
# val_sizeC = int(chinese.shape[0] * 0.15)
# test_sizeC = int(chinese.shape[0] * 0.15)
# test_chinese = chinese[:test_sizeC].reset_index(drop=True)
# val_chinese = chinese[test_sizeC:test_sizeC + val_sizeC].reset_index(drop=True)
# train_chinese = chinese[test_sizeC + val_sizeC:].reset_index(drop=True)
#
# ### english
#
# english = englishRaw[["sentence", "label"]].copy(deep=True).sample(frac=1).reset_index(drop=True)
# english.rename(columns={'sentence': 'text'}, inplace=True)
# english['language'] = 'english'
# english = english.sample(9000).reset_index(drop=True).copy()
# # print(english['label'].value_counts())
#
# val_sizeE = int(english.shape[0] * 0.15)
# test_sizeE = int(english.shape[0] * 0.15)
# test_english = english[:test_sizeE].reset_index(drop=True)
# val_english = english[test_sizeE:test_sizeE + val_sizeE].reset_index(drop=True)
# train_english = english[test_sizeE + val_sizeE:].reset_index(drop=True)
#
# train_condat = pd.concat([chinese[0:1], english[0:1], spanish[0:1], marathi[0:1], urdu[0:1]])
# # print(train_condat)



## Tibetan
## import
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
### TIbet

neg_data = pd.read_csv('D:/all_project/all_datasets/Tibet_sentiment/TSTDv2_negtive.txt.txt', names=["text"],header=None, sep='\t')
obj_data = pd.read_csv('D:/all_project/all_datasets/Tibet_sentiment/TSTDv2_objective.txt.txt', names=["text"],header=None, sep='\t')
pos_data = pd.read_csv('D:/all_project/all_datasets/Tibet_sentiment/TSTDv2_positive.txt.txt', names=["text"],header=None, sep='\t')

neg = neg_data[["text"]].copy(deep=True).sample(frac=1).reset_index(drop=True)
neg['label'] = -1

obj = obj_data[["text"]].copy(deep=True).sample(frac=1).reset_index(drop=True)
obj['label'] = 0

pos = pos_data[["text"]].copy(deep=True).sample(frac=1).reset_index(drop=True)
pos['label'] = 1
tibet = pd.concat([neg, pos]).sample(frac=1).reset_index(drop=True)
tibet['language'] = 'tibetan'
val_sizeT = int(tibet.shape[0]*0.15)
test_sizeT = int(tibet.shape[0]*0.15)
val_Tibet = tibet[:val_sizeT].reset_index(drop=True)
test_Tibet = tibet[val_sizeT:val_sizeT+test_sizeT].reset_index(drop=True)
train_Tibet = tibet[val_sizeT+test_sizeT:].reset_index(drop=True)
#
#
# ### combine and summary
# test = pd.concat([test_english, test_chinese, test_urdu, test_spanish, test_marathi]).sample(frac=1).reset_index(
#     drop=True)
# val = pd.concat([val_english, val_chinese, val_urdu, val_spanish, val_marathi]).sample(frac=1).reset_index(
#     drop=True)
# train = pd.concat([train_english, train_chinese, train_urdu, train_spanish, train_marathi]).sample(
#     frac=1).reset_index(drop=True)
#
# testSummary = pd.DataFrame(columns=test['language'].unique())
# temp = [0, 0, 0, 0, 0]
# testSummary.loc['positive'] = temp
# testSummary.loc['negative'] = temp
# testSummary.loc['total'] = temp
# for lang in test['language'].unique():
#     pos = test.loc[(test.language == lang)]['label'].value_counts()[1]
#     neg = test.loc[(test.language == lang)]['label'].value_counts()[0]
#     testSummary.loc['positive'][lang] = pos
#     testSummary.loc['negative'][lang] = neg
#     testSummary.loc['total'][lang] = pos + neg
# testSummary['total'] = testSummary.sum(axis=1)
# # print(testSummary)
#
# valSummary = pd.DataFrame(columns=val['language'].unique())
# temp = [0, 0, 0, 0, 0]
# valSummary.loc['positive'] = temp
# valSummary.loc['negative'] = temp
# valSummary.loc['total'] = temp
# for lang in val['language'].unique():
#     pos = val.loc[(val.language == lang)]['label'].value_counts()[1]
#     neg = val.loc[(val.language == lang)]['label'].value_counts()[0]
#     valSummary.loc['positive'][lang] = pos
#     valSummary.loc['negative'][lang] = neg
#     valSummary.loc['total'][lang] = pos + neg
# valSummary['total'] = valSummary.sum(axis=1)
# # print(valSummary)
#
# trainSummary = pd.DataFrame(columns=train['language'].unique())
# temp = [0, 0, 0, 0, 0]
# trainSummary.loc['positive'] = temp
# trainSummary.loc['negative'] = temp
# trainSummary.loc['total'] = temp
# for lang in train['language'].unique():
#     pos = train.loc[(train.language == lang)]['label'].value_counts()[1]
#     neg = train.loc[(train.language == lang)]['label'].value_counts()[0]
#     trainSummary.loc['positive'][lang] = pos
#     trainSummary.loc['negative'][lang] = neg
#     trainSummary.loc['total'][lang] = pos + neg
# trainSummary['total'] = trainSummary.sum(axis=1)
# # print(trainSummary)
#
# test_texts, test_labels = list(test.text), list(test.label)
# val_texts, val_labels = list(val.text), list(val.label)
# train_texts, train_labels = list(train.text), list(train.label)

# model setup
import os

# Parameters for fine-tuning
L_RATE = 3e-5
MAX_LEN = 20
NUM_EPOCHS = 3
BATCH_SIZE = 64
NUM_CORES = os.cpu_count()
# print(NUM_CORES)

model = XLMRobertaForSequenceClassification.from_pretrained(
    'xlm-roberta-base',
    num_labels=2,
# from_tf=True
)
model.to(device)


# Tokenization
class CompDataset(Dataset):

    def __init__(self, df):
        self.df_data = df

    def __getitem__(self, index):
        text = self.df_data.loc[index, 'text']
        # tokenization
        encoded_dict = rTokenizer.encode_plus(
            text,
            max_length=MAX_LEN,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        padded_token_list = encoded_dict['input_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        target = torch.tensor(self.df_data.loc[index, 'label'])
        sample = (padded_token_list, att_mask, target)
        return sample

    def __len__(self):
        return len(self.df_data)

# test_data = CompDataset(test)
# train_data = CompDataset(train)
# val_data = CompDataset(val)

# print(len(test_data))
# print(len(train_data))
# print(len(val_data))

## baseline result
def createBaseline(data, name):d
    test_data = CompDataset(data)
    dataloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True,
                                             num_workers=NUM_CORES)
    model.eval()
    torch.set_grad_enabled(False)
    targets_list = []
    for j, batch in enumerate(dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        outputs = model(b_input_ids,
                        attention_mask=b_input_mask,
                        labels=b_labels)
        loss = outputs[0]
        preds = outputs[1]
        val_preds = preds.detach().cpu().numpy()
        targets_np = b_labels.to('cpu').numpy()
        targets_list.extend(targets_np)
        if j == 0:
            stacked_val_preds = val_preds
        else:
            stacked_val_preds = np.vstack((stacked_val_preds, val_preds))
    y_true = targets_list
    y_pred = np.argmax(stacked_val_preds, axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(name, ' test acc: ', acc)
    print(name, ': test F1: ', f1)

import random
import gc

optimizer = AdamW(model.parameters(),
              lr = L_RATE,
              eps = 1e-8
            )

train_data = CompDataset(train_marathi)
val_data = CompDataset(val)

train_dataloader = torch.utils.data.DataLoader(train_data,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=NUM_CORES)
val_dataloader = torch.utils.data.DataLoader(val_data,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=NUM_CORES)


if __name__ == '__main__':
    freeze_support()
    # createBaseline(test_chinese, 'Chinese')
    # createBaseline(test_english, 'English')
    # createBaseline(test_urdu, 'Urdu')
    # createBaseline(test_marathi, 'Marathi')
    # createBaseline(test_spanish, 'Spanish')
    seed_val = 101

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    loss_values = []

    for epoch in range(0, NUM_EPOCHS):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, NUM_EPOCHS))

        stacked_val_labels = []
        targets_list = []

        print('Training...')
        model.train()
        torch.set_grad_enabled(True)
        total_train_loss = 0
        gc.collect()

        for i, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()
            outputs = model(b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs[0]
            total_train_loss = total_train_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            gc.collect()

        print('Train loss:', total_train_loss)

        gc.collect()

        print('\nValidation...')
        model.eval()
        torch.set_grad_enabled(False)
        total_val_loss = 0
        for j, batch in enumerate(val_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            outputs = model(b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs[0]
            total_val_loss = total_val_loss + loss.item()
            preds = outputs[1]
            val_preds = preds.detach().cpu().numpy()
            targets_np = b_labels.to('cpu').numpy()
            targets_list.extend(targets_np)
            if j == 0:  # first batch
                stacked_val_preds = val_preds
            else:
                stacked_val_preds = np.vstack((stacked_val_preds, val_preds))
            gc.collect()
        y_true = targets_list
        y_pred = np.argmax(stacked_val_preds, axis=1)
        val_acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print('Val loss:', total_val_loss)
        print('Val acc: ', val_acc)
        print('Val f1:', f1)
        gc.collect()

        torch.save(model.state_dict(), 'model.pt')

        gc.collect()



