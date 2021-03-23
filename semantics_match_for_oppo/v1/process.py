# -*- coding: utf-8 -*-
# @DateTime :2021/3/16
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
import datetime
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForPreTraining
from sklearn.model_selection import train_test_split

from v1 import readers, models, trainer
import settings as conf


def pretrain_model():
    # 训练数据
    train_path = f'{conf.DATA_ROOT}/train.tsv'
    valid_path = f'{conf.DATA_ROOT}/testA.tsv'
    train = pd.read_csv(train_path, sep='\t', names=['text_a', 'text_b', 'label'])
    testA = pd.read_csv(valid_path, sep='\t', names=['text_a', 'text_b', 'label'])
    testA['label'] = 2

    vocab = readers.build_vocab(train=train, testA=testA)  # 生成字典

    # 数据加载器
    train_dataset = readers.MlmNspDataset(train_path, vocab=vocab, seq_len=64)
    valid_dataset = readers.MlmNspDataset(valid_path, vocab=vocab, seq_len=64)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    # 定义模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = BertConfig(vocab_size=len(vocab) + 1)
    model = BertForPreTraining(config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # 训练模型
    model_root = f"{conf.MODEL_PATH}/pretrained"
    os.makedirs(model_root, exist_ok=True)
    model_path = os.path.join(model_root, f'pretrained_model.pth')
    trainer.pretrain_fit(model=model, train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, device=device, model_path=model_path)


def _fine_tune_fit(train_dataset, valid_dataset, model_path, bs=64):
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False)

    # 定义模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab = readers.load_vocab()  # 加载字典
    config = BertConfig(vocab_size=len(vocab) + 1)
    pretrained_model = BertForPreTraining(config)
    pretrained_model.load_state_dict(torch.load(f"{conf.MODEL_PATH}/pretrained/pretrained_model.pth", map_location=device))
    model = models.BertModel(pretrained_model, output_size=conf.NUM_CLASSES)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # 训练模型
    trainer.fine_tune_fit(model=model, train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, criterion=criterion, device=device, model_path=model_path)


def _fine_tune_predict(model_path, test, bs):
    vocab = readers.load_vocab()  # 加载字典
    test_dataset = readers.CustomDataset(test, word_dict=vocab, seq_len=64)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    # 定义模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = BertConfig(vocab_size=len(vocab) + 1)
    pretrained_model = BertForPreTraining(config)
    model = models.BertModel(pretrained_model, output_size=conf.NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    result = trainer.fine_tune_predict(model, dataloader=test_loader, device=device)
    return result


def fine_tune_model():
    # 训练数据
    train_path = f'{conf.DATA_ROOT}/train.tsv'
    train = pd.read_csv(train_path, sep='\t', names=['text_a', 'text_b', 'label'])
    # 拆分训练集和测试集
    train_index, valid_index = train_test_split(range(train.shape[0]), test_size=0.2)

    vocab = readers.load_vocab()  # 加载字典
    train_dataset = readers.CustomDataset(train.iloc[train_index], word_dict=vocab, seq_len=64)
    valid_dataset = readers.CustomDataset(train.iloc[valid_index], word_dict=vocab, seq_len=64)

    model_root = f"{conf.MODEL_PATH}/single"
    os.makedirs(model_root, exist_ok=True)
    model_path = os.path.join(model_root, 'fine_tune_model.pth')
    _fine_tune_fit(train_dataset=train_dataset, valid_dataset=valid_dataset, model_path=model_path)


def cv_fine_tune_fit():
    from sklearn.model_selection import StratifiedKFold
    model_root = f"{conf.MODEL_PATH}/{conf.K_FOLD}_folds"
    os.makedirs(model_root, exist_ok=True)

    # 训练数据
    train_path = f'{conf.DATA_ROOT}/train.tsv'
    train = pd.read_csv(train_path, sep='\t', names=['text_a', 'text_b', 'label'])
    vocab = readers.load_vocab()  # 加载字典
    t = train.label

    counter = 0
    skf = StratifiedKFold(n_splits=conf.K_FOLD, shuffle=True)
    for train_index, valid_index in skf.split(np.zeros(len(t)), t):
        train_dataset = readers.CustomDataset(train.iloc[train_index], word_dict=vocab, seq_len=64)
        valid_dataset = readers.CustomDataset(train.iloc[valid_index], word_dict=vocab, seq_len=64)

        model_path = os.path.join(model_root, f"fine_tune_model-{counter}.pth")
        _fine_tune_fit(train_dataset=train_dataset, valid_dataset=valid_dataset, model_path=model_path)

        counter += 1


def fine_tune_predict(cv=False, bs=64):
    if cv:
        model_root = f"{conf.MODEL_PATH}/{conf.K_FOLD}_folds"
    else:
        model_root = f"{conf.MODEL_PATH}/single"

    # 数据集
    test_path = f'{conf.DATA_ROOT}/testA.tsv'
    test = pd.read_csv(test_path, sep='\t', names=['text_a', 'text_b', 'label'])
    test['label'] = 2
    comb = test.copy(deep=True)

    model_paths = [os.path.join(model_root, fn) for fn in os.listdir(model_root) if fn.endswith('.pth')]

    for idx, model_path in enumerate(model_paths):
        result = _fine_tune_predict(model_path=model_path, test=test, bs=bs)
        comb[f"pred_{idx}"] = result[:, 1]

    now = datetime.datetime.now().strftime(r'%Y-%m-%d %H-%M-%S')

    cols = [fn for fn in test.columns.to_list() if fn.startswith('pred_')]
    comb['label'] = comb[cols].apply(lambda x: x.sum(), axis=1)
    comb.to_csv(f"{conf.DATA_PATH}/predict-all-{now}.csv", index=False)
    comb['label'] = comb['label'] / len(cols)
    comb['label'].to_csv(f"{conf.DATA_PATH}/predict-{now}.csv", sep='\t', index=0, header=False)


if __name__ == '__main__':
    # pretrain_model()
    fine_tune_model()
    fine_tune_predict(cv=False, bs=64)
