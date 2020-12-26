# -*- coding: utf-8 -*-
# @DateTime :2020/12/25 下午8:57
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import torch
from process import Config, train, convert_prediction_to_json
from datasets import MultiDatasetModule
from model_bert import ModelBert

conf = Config(
    model_name='bert',
    bert_pretrained=r'/Users/liuzhi/models/torch/bert-base-chinese',
    model_saved_path=r'./',
    log_path='./logs/',
    data_path=r'/Users/liuzhi/datasets/tc_nlp_generalizer',
    use_gpu=torch.cuda.is_available(),
    epochs=5,
    batch_size=4,
    lr=2e-5,
    classes_of_ocemotion=7,
    classes_of_ocnli=3,
    classes_of_tnews=15,
    fold=0,  # 使用第几折作为测试集
    max_len=64,
    num_workers=0,
)

if __name__ == '__main__':
    # train(conf, ModelBert, MultiDatasetModule)
    convert_prediction_to_json(conf)
