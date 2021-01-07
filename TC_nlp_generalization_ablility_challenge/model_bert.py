# -*- coding: utf-8 -*-
# @DateTime :2020/12/23 下午9:56
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from sklearn.metrics import accuracy_score
from tools import MultiFocalLoss


class ModelBert(pl.LightningModule):
    def __init__(self, params):
        super(ModelBert, self).__init__()
        self.params = params

        self.bert = AutoModel.from_pretrained(params.bert_pretrained)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.clf_ocemotion = nn.Linear(self.bert.config.hidden_size, params.classes_of_ocemotion)
        self.clf_ocnli = nn.Linear(self.bert.config.hidden_size, params.classes_of_ocnli)
        self.clf_tnews = nn.Linear(self.bert.config.hidden_size, params.classes_of_tnews)
        self.loss_fc = nn.Linear()

        # --- Focal Loss
        self.loss_ocemotion = MultiFocalLoss(params.classes_of_ocemotion, alpha=[4068, 4347, 590, 8894, 4042, 12475, 899])
        self.loss_ocnli = MultiFocalLoss(params.classes_of_ocnli, alpha=[16211, 16731, 15836])
        self.loss_tnews = MultiFocalLoss(params.classes_of_tnews, alpha=[1326, 4817, 5886, 4758, 6156, 2485, 4909, 4083, 7044, 4348, 4061, 5756, 302, 3380, 4049])

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
        # OCEMOTION
        ocemotion = batch['ocemotion']
        ocemotion_label = ocemotion['label']
        y_hat_ocemotion = self.forward_ocemotion(input_ids=ocemotion['input_ids'], attention_mask=ocemotion['attention_mask'])
        loss_ocemotion = self.loss_ocemotion(y_hat_ocemotion.view(-1, self.params.classes_of_ocemotion), ocemotion_label.view(-1))

        _, y_pred_ocemotion = torch.max(y_hat_ocemotion.view(-1, self.params.classes_of_ocemotion), dim=-1)
        acc_ocemotion = accuracy_score(y_pred_ocemotion.cpu(), ocemotion_label.cpu())
        self.log(f'{flag}_ocemotion_loss', loss_ocemotion)
        self.log(f'{flag}_ocemotion_accuracy', torch.tensor(acc_ocemotion))

        # OCNLI
        ocnli = batch['ocnli']
        ocnli_label = ocnli['label']
        y_hat_ocnli = self.forward_ocnli(input_ids=ocnli['input_ids'], attention_mask=ocnli['attention_mask'], token_type_ids=ocnli['token_type_ids'])
        loss_ocnli = self.loss_ocnli(y_hat_ocnli.view(-1, self.params.classes_of_ocnli), ocnli_label.view(-1))

        _, y_pred_ocnli = torch.max(y_hat_ocnli.view(-1, self.params.classes_of_ocnli), dim=-1)
        acc_ocnli = accuracy_score(y_pred_ocnli.cpu(), ocnli_label.cpu())
        self.log(f'{flag}_ocnli_loss', loss_ocnli)
        self.log(f'{flag}_ocnli_accuracy', torch.tensor(acc_ocnli))

        # TNEWS
        tnews = batch['tnews']
        tnews_label = tnews['label']
        y_hat_tnews = self.forward_tnews(input_ids=tnews['input_ids'], attention_mask=tnews['attention_mask'])
        loss_tnews = self.loss_tnews(y_hat_tnews.view(-1, self.params.classes_of_tnews), tnews_label.view(-1))

        _, y_pred_tnews = torch.max(y_hat_tnews.view(-1, self.params.classes_of_tnews), dim=-1)
        acc_tnews = accuracy_score(y_pred_tnews.cpu(), tnews_label.cpu())
        self.log(f'{flag}_tnews_loss', loss_tnews)
        self.log(f'{flag}_tnews_accuracy', torch.tensor(acc_tnews))

        loss = loss_ocemotion + loss_ocnli + loss_tnews
        acc = acc_ocemotion + acc_ocnli + acc_tnews
        self.log(f'{flag}_loss', loss_tnews)
        self.log(f'{flag}_accuracy', torch.tensor(acc))

        return loss

    def training_step(self, batch, batch_nb):
        loss = self._process_one_batch(batch, flag='train')
        return loss

    def validation_step(self, batch, batch_nb):
        return self._process_one_batch(batch, flag='valid')

    def test_step(self, batch, batch_nb):
        # OCEMOTION
        ocemotion = batch['ocemotion']
        y_hat_ocemotion = self.forward_ocemotion(input_ids=ocemotion['input_ids'], attention_mask=ocemotion['attention_mask'])
        _, y_pred_ocemotion = torch.max(y_hat_ocemotion.view(-1, self.params.classes_of_ocemotion), dim=-1)

        # OCNLI
        ocnli = batch['ocnli']
        y_hat_ocnli = self.forward_ocnli(input_ids=ocnli['input_ids'], attention_mask=ocnli['attention_mask'], token_type_ids=ocnli['token_type_ids'])
        _, y_pred_ocnli = torch.max(y_hat_ocnli.view(-1, self.params.classes_of_ocnli), dim=-1)

        # TNEWS
        tnews = batch['tnews']
        y_hat_tnews = self.forward_tnews(input_ids=tnews['input_ids'], attention_mask=tnews['attention_mask'])
        _, y_pred_tnews = torch.max(y_hat_tnews.view(-1, self.params.classes_of_tnews), dim=-1)

        return {'ocemotion': y_pred_ocemotion, 'ocnli': y_pred_ocnli, 'tnews': y_pred_tnews}

    def test_epoch_end(self, outputs):
        pred_ocemotion = torch.cat([output['ocemotion'] for output in outputs]).detach().cpu().numpy()
        pred_ocnli = torch.cat([output['ocnli'] for output in outputs]).detach().cpu().numpy()
        pred_tnews = torch.cat([output['tnews'] for output in outputs]).detach().cpu().numpy()
        self.save_prediction(ocemotion=pred_ocemotion, ocnli=pred_ocnli, tnews=pred_tnews)

    def save_prediction(self, ocemotion, ocnli, tnews):
        data_path = self.params.data_path
        test_tnews = pd.read_csv(f'{data_path}/test_tnews.csv')
        test_ocnli = pd.read_csv(f'{data_path}/test_ocnli.csv')
        test_ocemotion = pd.read_csv(f'{data_path}/test_ocemotion.csv')
        test_ocemotion['label'] = ocemotion
        test_ocnli['label'] = ocnli
        test_tnews['label'] = tnews

        test_ocemotion.to_csv(f'{data_path}/infer_ocemotion_{self.params.fold}.csv', index=False)
        test_ocnli.to_csv(f'{data_path}/infer_ocnli_{self.params.fold}.csv', index=False)
        test_tnews.to_csv(f'{data_path}/infer_tnews_{self.params.fold}.csv', index=False)
