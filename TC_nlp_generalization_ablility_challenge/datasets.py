# -*- coding: utf-8 -*-
# @DateTime :2020/12/23 下午9:02
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import pandas as pd
from torch.utils import data


class MutilDataset(data.Dataset):
    def __init__(self, ocemotion, ocnli, tnews, tokenizer, max_len=128):
        self.ocemotion = ocemotion[ocemotion.columns[1:]].values
        self.ocnli = ocnli[ocnli.columns[1:]].values
        self.tnews = tnews[tnews.columns[1:]].values

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.length = max(len(self.ocemotion), len(self.ocnli), len(self.tnews))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ocemotion = self._get_ocemotioin(idx)
        ocnli = self._get_ocnli(idx)
        tnews = self._get_tnews(idx)
        return {
            'ocemotion': ocemotion,
            'ocnli': ocnli,
            'tnews': tnews
        }

    def _get_ocemotioin(self, idx):
        item = self.ocemotion[idx % len(self.ocemotion)]
        return self._encoding(text=item[0], label=item[-1])

    def _get_ocnli(self, idx):
        item = self.ocnli[idx % len(self.ocnli)]
        return self._encoding(text=item[0], text_pair=item[1], label=item[-1])

    def _get_tnews(self, idx):
        item = self.tnews[idx % len(self.tnews)]
        return self._encoding(text=item[0], label=item[-1])

    def _encoding(self, text, text_pair=None, label=None):
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

        encoding = {
            'text': text,
            # 'label': torch.tensor(label, dtype=torch.long),
            'input_ids': (encoding['input_ids']).flatten(),
            'attention_mask': (encoding['attention_mask']).flatten(),
            'token_type_ids': (encoding['token_type_ids']).flatten()
        }

        if text_pair is not None:
            encoding['text_pair'] = text_pair

        return encoding


if __name__ == '__main__':
    from transformers import BertTokenizer

    root_data = r'/Users/liuzhi/datasets/tc_nlp_generalizer'

    tokenizer = BertTokenizer.from_pretrained(r'/Users/liuzhi/models/torch/bert-base-chinese')
    df_OCEMOTION = pd.read_csv(f"{root_data}/OCEMOTION_train1128.csv", header=None, sep='\t')  # 35k
    df_OCNLI = pd.read_csv(f"{root_data}/OCNLI_train1128.csv", header=None, sep='\t')  # 48k
    df_TNEWS = pd.read_csv(f"{root_data}/TNEWS_train1128.csv", header=None, sep='\t')  # 63k
    dset = MutilDataset(df_OCEMOTION, df_OCNLI, df_TNEWS, tokenizer)
    for d in dset:
        print(d)
        break
