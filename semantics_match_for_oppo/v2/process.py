# -*- coding: utf-8 -*-
# @DateTime :2021/3/19
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
import datetime
import torch
import pandas as pd
from transformers import BertModel
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import settings as conf
from v2 import readers, models, trainer


def save_bert_embeddings(model_path, model_name="bert-base-chinese"):
    if not os.path.exists(model_path):
        model = BertModel.from_pretrained(model_name)
        torch.save(model.embeddings.word_embeddings.state_dict(), model_path)


def pretrain_model():
    # 训练数据
    train_path = f'{conf.DATA_ROOT}/train.tsv'
    valid_path = f'{conf.DATA_ROOT}/testA.tsv'
    train = pd.read_csv(train_path, sep='\t', names=['text_a', 'text_b', 'label'])
    testA = pd.read_csv(valid_path, sep='\t', names=['text_a', 'text_b', 'label'])
    testA['label'] = 2

    readers.build_vocab(train, testA, min_count=5)
    vocab, keep_tokens = readers.load_vocab_and_align_with_bert()

    train_index, valid_index = train_test_split(range(train.shape[0]), test_size=0.1, random_state=2021)
    train_data = train.iloc[train_index]
    valid_data = train.iloc[valid_index]
    train_dataset = readers.MlmNspDataset(train_data, vocab=vocab, seq_len=64)
    valid_dataset = readers.MlmNspDataset(valid_data, vocab=vocab, seq_len=64)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    # 定义模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = f"{conf.DATA_ROOT}/bert_embeddings.pth"
    readers.save_bert_embeddings(checkpoint_path)
    model = models.PretrainedBertModel(embeddings_size=len(vocab), checkpoint_path=checkpoint_path, keep_tokens=keep_tokens)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    model_root = os.path.join(conf.MODEL_PATH, 'pretrained')
    os.makedirs(model_root, exist_ok=True)
    model_path = os.path.join(model_root, f'pretrained_bert.pth')
    trainer.pretrain_fit(model=model, train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, criterion=criterion, vocab=vocab, device=device, model_path=model_path)


def predict():
    # 训练数据
    valid_path = f'{conf.DATA_ROOT}/testA.tsv'
    testA = pd.read_csv(valid_path, sep='\t', names=['text_a', 'text_b', 'label'])
    testA['label'] = 2

    vocab, keep_tokens = readers.load_vocab_and_align_with_bert()
    test_dataset = readers.MlmNspDataset(testA, vocab=vocab, seq_len=64)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 定义模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = f"{conf.DATA_ROOT}/bert_embeddings.pth"
    model_root = os.path.join(conf.MODEL_PATH, 'pretrained')
    model_path = os.path.join(model_root, f'pretrained_bert.pth')
    model = models.PretrainedBertModel(embeddings_size=len(vocab), checkpoint_path=checkpoint_path, keep_tokens=keep_tokens)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    result = trainer.pretrain_predict(model, dataloader=test_loader, device=device)

    now = datetime.datetime.now().strftime(r'%Y-%m-%d %H-%M-%S')
    testA[f"label"] = result
    testA.to_csv(f"{conf.DATA_PATH}/predict-all-{now}.csv", index=False)


if __name__ == '__main__':
    save_bert_embeddings(model_path=os.path.join(conf.MODEL_PATH, 'bert_embeddings.pth'))
