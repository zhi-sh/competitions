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


def train(conf, module):
    model = module(conf)

    # 使用GPU配置
    gpu_num = 1 if conf.use_gpu else None
    precision = 16 if conf.use_gpu else 32  # 默认精度32位，使用GPU时16位

    tb_logger = pl.loggers.TensorBoardLogger(conf.log_path)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=conf.model_saved_path,
        filename=conf.model_name,
        monitor=r'valid_loss',  # 依据验证集损失
        mode='min',
        verbose=False,
    )

    trainer = pl.Trainer(
        gpus=gpu_num,
        precision=precision,
        max_epochs=conf.epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback]
    )

    # 训练+测试
    trainer.fit(model)
    trainer.test(ckpt_path=trainer.checkpoint_callback.best_model_path)
