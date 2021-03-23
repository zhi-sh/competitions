# -*- coding: utf-8 -*-
# @DateTime :2021/3/16
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score


# ------------------------ 预训练 ----------------------------
def pretrain_evaluate(model, dataloader, device):
    model.eval()
    losses = []
    for data, is_nsp in tqdm(dataloader):
        is_nsp = is_nsp.to(device).long()
        input_ids = data['input_ids'].to(device).long()
        token_type_ids = data['token_type_ids'].to(device).long()
        attention_mask = data['attention_mask'].to(device).long()
        labels = data['bert_label'].to(device).long()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, next_sentence_label=is_nsp)
            loss = outputs['loss']
            losses.append(loss.cpu().detach().numpy())

    loss = np.mean(losses)
    return loss


def pretrain_fit(model, train_loader, valid_loader, optimizer, device, model_path):
    min_loss = 1e9
    round_counter = 0
    early_stopped_thresh = 5

    for epoch in range(10000):
        model.train()
        losses = []
        progress_bar = tqdm(train_loader)
        for data, is_nsp in progress_bar:
            is_nsp = is_nsp.to(device).long()
            input_ids = data['input_ids'].to(device).long()
            token_type_ids = data['token_type_ids'].to(device).long()
            attention_mask = data['attention_mask'].to(device).long()
            labels = data['bert_label'].to(device).long()

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, next_sentence_label=is_nsp)
            loss = outputs['loss']
            losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Pretrained model at Epoch: {epoch} train loss: {np.mean(losses)}")

        eval_loss = pretrain_evaluate(model=model, dataloader=valid_loader, device=device)
        if eval_loss < min_loss:
            min_loss = eval_loss
            round_counter = 0
            torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)
            print(f"Pretrained model at Epoch: {epoch}, train loss: {np.mean(losses)}, valid loss: {eval_loss}")

        round_counter += 1
        if round_counter > early_stopped_thresh:
            break


# ------------------------ 微调模型 ----------------------------
def fine_tune_evaluate(model, dataloader, criterion, device):
    model.eval()
    losses = []
    labels_list = []
    preds_list = []
    for data, label in tqdm(dataloader):
        label = label.to(device).long()
        input_ids = data['input_ids'].to(device).long()
        token_type_ids = data['token_type_ids'].to(device).long()
        attention_mask = data['attention_mask'].to(device).long()

        with torch.no_grad():
            preds = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            loss = criterion(preds, label)
            preds = torch.softmax(preds, dim=-1)
            preds_list.append(preds.argmax(-1).cpu().detach().numpy())
            labels_list.append(label.cpu().detach().numpy())
            losses.append(loss.cpu().detach().item())
    results = np.concatenate(preds_list)
    labels = np.concatenate(labels_list)
    accuracy = accuracy_score(labels, results)
    eval_loss = sum(losses) / len(dataloader.dataset)
    return accuracy, eval_loss


def fine_tune_fit(model, train_loader, valid_loader, optimizer, criterion, device, model_path):
    min_loss = 1e9
    round_counter = 0
    early_stopped_thresh = 5

    for epoch in range(10000):
        model.train()
        losses = []
        progress_bar = tqdm(train_loader)
        for data, label in progress_bar:
            label = label.to(device).long()
            input_ids = data['input_ids'].to(device).long()
            token_type_ids = data['token_type_ids'].to(device).long()
            attention_mask = data['attention_mask'].to(device).long()

            optimizer.zero_grad()
            preds = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
            progress_bar.set_description(f"Fine tune model at Epoch: {epoch} train loss: {np.mean(losses)}")

        eval_acc, eval_loss = fine_tune_evaluate(model=model, dataloader=valid_loader, criterion=criterion, device=device)
        if eval_loss < min_loss:
            min_loss = eval_loss
            round_counter = 0
            torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)
            print(f"Fine tune model at Epoch: {epoch}, train loss: {np.mean(losses)}, valid loss: {eval_loss}")

        round_counter += 1
        if round_counter > early_stopped_thresh:
            break


def fine_tune_predict(model, dataloader, device):
    model.eval()
    preds_list = []
    progress_bar = tqdm(dataloader)
    for data, _ in progress_bar:
        input_ids = data['input_ids'].to(device).long()
        token_type_ids = data['token_type_ids'].to(device).long()
        attention_mask = data['attention_mask'].to(device).long()

        with torch.no_grad():
            preds = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            preds = torch.softmax(preds, dim=-1)
            preds_list.append(preds.cpu().detach().numpy())

    preds = np.concatenate(preds_list)
    return preds
