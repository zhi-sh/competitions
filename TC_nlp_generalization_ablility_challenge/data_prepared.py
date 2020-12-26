# -*- coding: utf-8 -*-
# @DateTime :2020/12/25 下午8:34
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import pickle
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold


def load_weights(lb_path, df_path, y):
    le = pickle.load(open(lb_path, 'rb'))
    df = pd.read_csv(df_path)
    vals_dict = df[y].value_counts().to_dict()
    labels = le.inverse_transform(range(len(vals_dict)))
    weights = [vals_dict[i] for i in labels]
    return weights


def label_encoding(df, fpath):
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(df[['target']].values.ravel())
    df['label'] = y.reshape(-1)
    pickle.dump(le, open(fpath, 'wb'))


def startify_and_save(df, saved_path, K=10):
    r'''分层采样'''
    columns = df.columns.to_list()
    if 'text_pair' in columns:
        xcol = ['text', 'text_pair']
    else:
        xcol = ['text']

    df.loc[:, 'kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    sfolder = StratifiedKFold(n_splits=K)
    for fold_idx, (trn_, val_) in enumerate(sfolder.split(X=df[xcol], y=df['label'])):
        df.loc[val_, 'kfold'] = fold_idx
    df.to_csv(saved_path, index=False)


def preparing_train(conf):
    data_path = conf.data_path

    df_ocemotion = pd.read_csv(f'{data_path}/OCEMOTION_train1128.csv', header=None, sep='\t')
    df_ocemotion['text'] = df_ocemotion[df_ocemotion.columns[1]]
    df_ocemotion['target'] = df_ocemotion[df_ocemotion.columns[2]]
    label_encoding(df_ocemotion, fpath=f'{data_path}/lb_ocemotion.pkl')
    startify_and_save(df_ocemotion[['text', 'target', 'label']], f'{data_path}/train_ocemotion.csv')

    df_tnews = pd.read_csv(f'{data_path}/TNEWS_train1128.csv', header=None, sep='\t')
    df_tnews['text'] = df_tnews[df_tnews.columns[1]]
    df_tnews['target'] = df_tnews[df_tnews.columns[2]]
    label_encoding(df_tnews, fpath=f'{data_path}/lb_tnews.pkl')
    startify_and_save(df_tnews[['text', 'target', 'label']], f'{data_path}/train_tnews.csv')

    df_ocnli = pd.read_csv(f'{data_path}/OCNLI_train1128.csv', header=None, sep='\t')
    df_ocnli['text'] = df_ocnli[df_ocnli.columns[1]]
    df_ocnli['text_pair'] = df_ocnli[df_ocnli.columns[2]]
    df_ocnli['target'] = df_ocnli[df_ocnli.columns[3]]
    label_encoding(df_ocnli, fpath=f'{data_path}/lb_ocnli.pkl')
    startify_and_save(df_ocnli[['text', 'text_pair', 'target', 'label']], f'{data_path}/train_ocnli.csv')

    ocemotion_weights = load_weights(lb_path=f'{data_path}/lb_ocemotion.pkl', df_path=f'{data_path}/train_ocemotion.csv', y='target')
    tnews_weights = load_weights(lb_path=f'{data_path}/lb_tnews.pkl', df_path=f'{data_path}/train_tnews.csv', y='target')
    ocnli_weights = load_weights(lb_path=f'{data_path}/lb_ocnli.pkl', df_path=f'{data_path}/train_ocnli.csv', y='target')

    print(f'ocemotion : {ocemotion_weights}')
    print(f'tnews : {tnews_weights}')
    print(f'ocnli : {ocnli_weights}')


def preparing_test(conf):
    data_path = conf.data_path

    df_ocemotion = pd.read_csv(f'{data_path}/OCEMOTION_a.csv', header=None, sep='\t')
    df_ocemotion['id'] = df_ocemotion[df_ocemotion.columns[0]]
    df_ocemotion['text'] = df_ocemotion[df_ocemotion.columns[1]]
    df_ocemotion[['id', 'text']].to_csv(f'{data_path}/test_ocemotion.csv', index=False)

    df_tnews = pd.read_csv(f'{data_path}/TNEWS_a.csv', header=None, sep='\t')
    df_tnews['id'] = df_tnews[df_tnews.columns[0]]
    df_tnews['text'] = df_tnews[df_tnews.columns[1]]
    df_tnews[['id', 'text']].to_csv(f'{data_path}/test_tnews.csv', index=False)

    df_ocnli = pd.read_csv(f'{data_path}/OCNLI_a.csv', header=None, sep='\t')
    df_ocnli['id'] = df_ocnli[df_ocnli.columns[0]]
    df_ocnli['text'] = df_ocnli[df_ocnli.columns[1]]
    df_ocnli['text_pair'] = df_ocnli[df_ocnli.columns[2]]
    df_ocnli[['id', 'text', 'text_pair']].to_csv(f'{data_path}/test_ocnli.csv', index=False)
    print('test info ...')
    print(df_ocemotion.shape, df_tnews.shape, df_ocnli.shape)


if __name__ == '__main__':
    from process import Config

    conf = Config(
        data_path=r'/Users/liuzhi/datasets/tc_nlp_generalizer'
    )

    # prepare train data
    preparing_train(conf)

    # prepare test_data
    preparing_test(conf)
