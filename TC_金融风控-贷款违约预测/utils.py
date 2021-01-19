# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


# 求熵
def myEntro(x):
    """
        calculate shanno ent of x
    """
    x = np.array(x)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    #     print(x_value,p,logp)
    # print(ent)
    return ent


# 均方根
def myRms(records):
    records = list(records)
    """
    均方根值 反映的是有效值而不是平均值
    """
    return np.math.sqrt(sum([x**2 for x in records]) / len(records))


# 众数
def myMode(x):
    return np.mean(pd.Series.mode(x))


# 值的范围
def myRange(x):
    return pd.Series.max(x) - pd.Series.min(x)


# 分别求取10，25，75，90分位值
def myQ25(x):
    return x.quantile(0.25)


def myQ75(x):
    return x.quantile(0.75)


def myQ10(x):
    return x.quantile(0.1)


def myQ90(x):
    return x.quantile(0.9)


# reduce_mem_usage 函数通过调整数据类型，帮助我们减少数据在内存中占用的空间
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                        np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) /
                                        start_mem))
    return df


# 拆分数字特征和离散特征列表
def num_cat_cols(x, THRESH=50):
    a = x.apply(lambda x: x.nunique())
    num_cols = a[a > THRESH].index.tolist()
    cat_cols = a[a <= THRESH].index.tolist()
    return num_cols, cat_cols