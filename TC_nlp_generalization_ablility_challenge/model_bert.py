# -*- coding: utf-8 -*-
# @DateTime :2020/12/23 下午9:56
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score

from datasets import MutilDataset


class ModelBert(pl.LightningModule):
    def __init__(self, params):
        super(ModelBert, self).__init__()
        self.params = params
        self.tokenizer = AutoTokenizer.from_pretrained(params.bert_pretrained)
        self.bert = AutoModel.from_pretrained(params.bert_pretrained)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.clf_ocemotion = nn.Linear(self.bert.config.hidden_size, params.classes_of_ocemotion)
        self.clf_ocnli = nn.Linear(self.bert.config.hidden_size, params.classes_of_ocnli)
        self.clf_tnews = nn.Linear(self.bert.config.hidden_size, params.classes_of_tnews)

    def _forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        return pooled_output

    def forward_ocemotion(self, input_ids, attention_mask):
        pooled_output = self._forward(input_ids, attention_mask)
        return self.clf_ocemotion(pooled_output)

    def forward_ocnli(self, input_ids, attention_mask, token_type_ids):
        pooled_output = self._forward(input_ids, attention_mask, token_type_ids)
        return self.clf_ocnli(pooled_output)

    def forward_tnews(self, input_ids, attention_mask):
        pooled_output = self._forward(input_ids, attention_mask)
        return self.clf_tnews(pooled_output)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.lr)
        return optimizer

    # --------------- Train + Valid + Test ------------
    def _process_one_batch(self, batch, flag='train'):
        loss_func = nn.CrossEntropyLoss()

        # OCEMOTION
        ocemotion = batch['ocemotion']
        ocemotion_label = ocemotion['label']
        y_hat_ocemotion = self.forward_ocemotion(input_ids=ocemotion['input_ids'], attention_mask=ocemotion['attention_mask'])
        loss_ocemotion = loss_func(y_hat_ocemotion.view(-1, self.params.classes_of_ocemotion), ocemotion.view(-1))

        _, y_pred_ocemotion = torch.max(y_hat_ocemotion.view(-1, self.params.classes_of_ocemotion), dim=-1)
        acc_ocemotion = accuracy_score(y_pred_ocemotion.cpu(), ocemotion_label.cpu())
        self.log(f'{flag}_ocemotion_loss', loss_ocemotion)
        self.log(f'{flag}_ocemotion_accuracy', torch.tensor(acc_ocemotion))

        # OCNLI
        ocnli = batch['ocnli']
        ocnli_label = ocnli['label']
        y_hat_ocnli = self.forward_ocnli(input_ids=ocnli['input_ids'], attention_mask=ocnli['attention_mask'])
        loss_ocnli = loss_func(y_hat_ocnli.view(-1, self.params.classes_of_ocnli), ocnli.view(-1))

        _, y_pred_ocnli = torch.max(y_hat_ocnli.view(-1, self.params.classes_of_ocnli), dim=-1)
        acc_ocnli = accuracy_score(y_pred_ocnli.cpu(), ocnli_label.cpu())
        self.log(f'{flag}_ocnli_loss', loss_ocnli)
        self.log(f'{flag}_ocnli_accuracy', torch.tensor(acc_ocnli))

        # TNEWS
        tnews = batch['tnews']
        tnews_label = tnews['label']
        y_hat_tnews = self.forward_tnews(input_ids=tnews['input_ids'], attention_mask=tnews['attention_mask'])
        loss_tnews = loss_func(y_hat_tnews.view(-1, self.params.classes_of_tnews), tnews.view(-1))

        _, y_pred_tnews = torch.max(y_hat_tnews.view(-1, self.params.classes_of_tnews), dim=-1)
        acc_tnews = accuracy_score(y_pred_tnews.cpu(), tnews_label.cpu())
        self.log(f'{flag}_tnews_loss', loss_tnews)
        self.log(f'{flag}_tnews_accuracy', torch.tensor(acc_tnews))

        loss = loss_ocemotion + loss_ocnli + loss_tnews
        acc = acc_ocemotion + acc_ocnli + acc_tnews
        self.log(f'{flag}_total_loss', loss_tnews)
        self.log(f'{flag}_total_accuracy', torch.tensor(acc))

        return loss

    def training_step(self, batch, batch_nb):
        loss = self._process_one_batch(batch, flag='train')
        return loss

    def validation_step(self, batch, batch_nb):
        return self._process_one_batch(batch, flag='val')

    def test_step(self, batch, batch_nb):
        return self._process_one_batch(batch, flag='test')

    # ------------------- 以下加载数据 ------------------
    def prepare_data(self):
        root_data = self.params.data_path
        df_OCEMOTION = pd.read_csv(f"{root_data}/OCEMOTION_train1128.csv", header=None, sep='\t')  # 35k
        df_OCNLI = pd.read_csv(f"{root_data}/OCNLI_train1128.csv", header=None, sep='\t')  # 48k
        df_TNEWS = pd.read_csv(f"{root_data}/TNEWS_train1128.csv", header=None, sep='\t')  # 63k

        self.train_set = MutilDataset(df_OCEMOTION, df_OCNLI, df_TNEWS, self.tokenizer)

        self.val_set = MutilDataset(df_OCEMOTION.sample(2000), df_OCNLI.sample(2000), df_TNEWS.sample(2000), self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.params.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.params.batch_size, shuffle=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.params.batch_size, shuffle=True, num_workers=4)


if __name__ == '__main__':
    from common import Config, train

    conf = Config(
        model_name='bert',

        bert_pretrained=r'/Users/liuzhi/models/torch/bert-base-chinese',
        model_saved_path=r'./',
        log_path='./logs/',
        data_path=r'/Users/liuzhi/datasets/tc_nlp_generalizer',
        use_gpu=torch.cuda.is_available(),
        epochs=10,
        batch_size=16,
        lr=2e-5,
        classes_of_ocemotion=2,
        classes_of_ocnli=3,
        classes_of_tnews=15
    )

    train(conf, ModelBert)
