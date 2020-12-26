# -*- coding: utf-8 -*-
# @DateTime :2020/12/23 下午9:02
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import pickle
import pandas as pd
import torch
import transformers
import pytorch_lightning as pl
from torch.utils import data


class MutilDataset(data.Dataset):
    def __init__(self, tnews, ocnli, ocemotion, tokenizer, max_len=128, is_test=False):
        self.tnews = tnews
        self.ocnli = ocnli
        self.ocemotion = ocemotion

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.length = max(len(self.ocemotion), len(self.ocnli), len(self.tnews))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ocemotion = self._get_single(self.ocemotion, idx)
        tnews = self._get_single(self.tnews, idx)
        ocnli = self._get_couple(self.ocnli, idx)
        return {
            'ocemotion': ocemotion,
            'ocnli': ocnli,
            'tnews': tnews
        }

    def _get_single(self, data, idx):
        item = data[idx % len(data)]
        label = 0 if self.is_test else item[-1]
        return self._encoding(text=item[0], label=label)

    def _get_couple(self, data, idx):
        item = data[idx % len(data)]
        label = 0 if self.is_test else item[-1]
        return self._encoding(text=item[0], label=label, text_pair=item[1])

    def _encoding(self, text, label=None, text_pair=None):
        encoding = self.tokenizer.encode_plus(
            text,
            text_pair,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'label': torch.tensor(label, dtype=torch.long),
            'text_pair': text_pair if text_pair else '',
            'input_ids': (encoding['input_ids']).flatten(),
            'attention_mask': (encoding['attention_mask']).flatten(),
            'token_type_ids': (encoding['token_type_ids']).flatten()
        }


class MultiDatasetModule(pl.LightningDataModule):
    def __init__(self, params, kfold=5):
        super(MultiDatasetModule, self).__init__()
        self.params = params
        self.kfold = kfold
        self.fold = params.fold
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(params.bert_pretrained)

    def prepare_data(self):
        data_path = self.params.data_path
        self.tnews = pd.read_csv(f'{data_path}/train_tnews.csv')
        self.ocnli = pd.read_csv(f'{data_path}/train_ocnli.csv')
        self.ocemotion = pd.read_csv(f'{data_path}/train_ocemotion.csv')
        self.test_tnews = pd.read_csv(f'{data_path}/test_tnews.csv')
        self.test_ocnli = pd.read_csv(f'{data_path}/test_ocnli.csv')
        self.test_ocemotion = pd.read_csv(f'{data_path}/test_ocemotion.csv')

    def setup(self, stage=None):
        tnews_train, tnews_val = self._train_valid_split(self.tnews)
        ocnli_train, ocnli_val = self._train_valid_split(self.ocnli)
        ocemotion_train, ocemotion_val = self._train_valid_split(self.ocemotion)
        self.train_dataset = MutilDataset(
            tnews_train[['text', 'label']].values,
            ocnli_train[['text', 'text_pair', 'label']].values,
            ocemotion_train[['text', 'label']].values,
            self.tokenizer,
            self.params.max_len
        )
        self.valid_dataset = MutilDataset(
            tnews_val[['text', 'label']].values,
            ocnli_val[['text', 'text_pair', 'label']].values,
            ocemotion_val[['text', 'label']].values,
            self.tokenizer,
            self.params.max_len
        )
        self.test_dataset = MutilDataset(
            self.test_tnews[['text']].values,
            self.test_ocnli[['text', 'text_pair']].values,
            self.test_ocemotion[['text']].values,
            self.tokenizer,
            self.params.max_len,
            is_test=True
        )

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.params.batch_size, num_workers=self.params.num_workers, shuffle=True)

    def val_dataloader(self):
        return data.DataLoader(self.valid_dataset, batch_size=self.params.batch_size, num_workers=self.params.num_workers, shuffle=False)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.params.batch_size, num_workers=self.params.num_workers, shuffle=False)

    def _train_valid_split(self, data):
        training_data = data[data.kfold != self.fold]
        training_data.drop(['kfold'], axis=1)
        validation_data = data[data.kfold == self.fold]
        validation_data.drop(['kfold'], axis=1)
        return training_data, validation_data


if __name__ == '__main__':
    from process import Config

    conf = Config(
        model_name='bert',
        bert_pretrained=r'/Users/liuzhi/models/torch/bert-base-chinese',
        model_saved_path=r'./',
        log_path='./logs/',
        data_path=r'/Users/liuzhi/datasets/tc_nlp_generalizer',
        use_gpu=torch.cuda.is_available(),
        epochs=10,
        batch_size=1,
        lr=2e-5,
        classes_of_ocemotion=2,
        classes_of_ocnli=3,
        classes_of_tnews=15,
        fold=0,
        max_len=64,
        num_workers=0,
    )

    mdm = MultiDatasetModule(conf)
    mdm.prepare_data()
    mdm.setup()
    loader = mdm.test_dataloader()
    for d in loader:
        print(d)
        break
