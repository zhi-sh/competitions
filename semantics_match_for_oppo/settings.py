# -*- coding: utf-8 -*-
# @DateTime :2021/3/16
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
import random
import numpy as np
import torch


def seed_everything(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


VERSION = 'v2'

NUM_CLASSES = 2  # 分类的数量

DATA_ROOT = f"data"
DATA_PATH = f"data/{VERSION}"
MODEL_PATH = f'data/{VERSION}/models'
K_FOLD = 5

# 预操作
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
seed_everything()
