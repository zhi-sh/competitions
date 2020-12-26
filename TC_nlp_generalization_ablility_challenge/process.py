# -*- coding: utf-8 -*-
# @DateTime :2020/12/23 下午6:11
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import pytorch_lightning as pl


class Config(dict):
    r'''
        全局配置函数
    '''

    def __init__(self, **kwargs):
        super(Config, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, k, v):
        self[k] = v
        setattr(self, k, v)


def train(conf, module, data_module):
    model = module(conf)
    dm = data_module(conf)

    tb_logger = pl.loggers.TensorBoardLogger(conf.log_path)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=conf.model_saved_path,
        filename=conf.model_name,
        monitor='valid_loss',
        mode='min',
        verbose=False,
    )

    trainer = pl.Trainer(
        gpus=-1 if conf.use_gpu else None,
        precision=16 if conf.use_gpu else 32,
        max_epochs=conf.epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, dm)
    trainer.test(ckpt_path=trainer.checkpoint_callback.best_model_path)


def inverse_label(df, le_path):
    import pickle

    le = pickle.load(open(le_path, 'rb'))
    y_inv = le.inverse_transform(df['label'])
    df['target'] = y_inv


def convert_prediction_to_json(conf):
    import json
    import pandas as pd
    jobs = ['ocemotion', 'ocnli', 'tnews']
    for job in jobs:
        df = pd.read_csv(f'{conf.data_path}/infer_{job}_0.csv')
        inverse_label(df, f'{conf.data_path}/lb_{job}.pkl')
        with open(f'{conf.data_path}/{job}_predict.json', 'w', encoding='utf-8') as fw:
            for ix, row in df.iterrows():
                em = {'id': str(row['id']), 'label': str(row['target'])}
                print(em)
                fw.write(f"{json.dumps(em, ensure_ascii=False)}\n")
