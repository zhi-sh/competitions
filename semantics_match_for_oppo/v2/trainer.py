# -*- coding: utf-8 -*-
# @DateTime :2021/3/19
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


# ------------------------ 预训练 ----------------------------
def pretrain_evaluate(model, dataloader, criterion, vocab, device):
    model.eval()
    losses = []
    labels_list = []
    preds_list = []
    progress_bar = tqdm(dataloader)
    data_dict = {}
    for data, is_nsp in progress_bar:
        data_dict['input_ids'] = data['input_ids'].to(device).long()
        data_dict['token_type_ids'] = data['token_type_ids'].to(device).long()
        data_dict['attention_mask'] = data['attention_mask'].to(device).long()
        labels = data['bert_label'].to(device).long()

        with torch.no_grad():
            outputs = model(data_dict)
            mask = (labels != -100)
            loss = criterion(outputs[mask].view(-1, len(vocab)), labels[mask].view(-1))
            losses.append(loss.cpu().detach().numpy())
            # 取出第6个和第7个tokens，因为这个是预留的两个token作为yes 和 no
            preds = outputs[:, 0, 5:7].cpu().detach().numpy()
            preds = preds[:, 1] / (preds.sum(axis=1) + 1e-8)
            labels = labels[:, 0] - 5
            labels_list.append(labels.cpu().detach().numpy())
            preds_list.append(preds)
    labels_list = np.concatenate(labels_list)
    preds_list = np.concatenate(preds_list)
    auc_score = roc_auc_score(labels_list, preds_list)
    return auc_score, np.mean(losses)


def pretrain_fit(model, train_loader, valid_loader, optimizer, criterion, vocab, device, model_path):
    min_loss = 1e9
    round_counter = 0
    early_stopped_thresh = 10

    for epoch in range(10000):
        model.train()
        losses = []
        progress_bar = tqdm(train_loader)
        data_dict = {}
        for data, is_nsp in progress_bar:
            data_dict['input_ids'] = data['input_ids'].to(device).long()
            data_dict['token_type_ids'] = data['token_type_ids'].to(device).long()
            data_dict['attention_mask'] = data['attention_mask'].to(device).long()
            labels = data['bert_label'].to(device).long()

            optimizer.zero_grad()
            outputs = model(data_dict)
            mask = (labels != -100)
            loss = criterion(outputs[mask].view(-1, len(vocab)), labels[mask].view(-1))
            losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Pretrained model at Epoch: {epoch} train loss: {np.mean(losses)}")

        eval_score, eval_loss = pretrain_evaluate(model=model, dataloader=valid_loader, criterion=criterion, vocab=vocab, device=device)
        if eval_loss < min_loss:
            min_loss = eval_loss
            round_counter = 0
            torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)
            print(f"Pretrained model at Epoch: {epoch}, train loss: {np.mean(losses)}, valid loss: {eval_loss}, valid score: {eval_score}")

        round_counter += 1
        if round_counter > early_stopped_thresh:
            break


def pretrain_predict(model, dataloader, device):
    model.eval()
    preds_list = []
    progress_bar = tqdm(dataloader)
    data_dict = {}
    for data, _ in progress_bar:
        data_dict['input_ids'] = data['input_ids'].to(device).long()
        data_dict['token_type_ids'] = data['token_type_ids'].to(device).long()
        data_dict['attention_mask'] = data['attention_mask'].to(device).long()

        with torch.no_grad():
            outputs = model(data_dict)
            preds = outputs[:, 0, 5:7].cpu().detach().numpy()
            preds = preds[:, 1] / (preds.sum(axis=1) + 1e-8)
            preds_list.append(preds)

    results = np.concatenate(preds_list)
    return results
