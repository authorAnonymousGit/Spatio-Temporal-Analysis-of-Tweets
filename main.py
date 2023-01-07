import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from collections import defaultdict
import Process_Data
import pandas as pd
import ACRCNN_train
import ACRCNN_test
from config import ConfigMain
from config import ConfigPrimary

import random
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_vocab(dfs_list, feature_text):
    word_dict = defaultdict(lambda: 0)
    for df in dfs_list:
        sentences = df[feature_text]
        for sentence in sentences:
            splitted_text = sentence.split(' ')
            for word in splitted_text:
                word_dict[word] += 1
    return word_dict


def write_to_file(file_num, text):
    print(text)
    fout = open(str(file_num) + ".txt", "a")
    fout.write(text + '\n')
    fout.close()


def preprocessing(df, word_dict, language, task=None):
    data_reader = Process_Data.TweetsDataReader(df, word_dict, language, task)
    return data_reader


def read_config_primary():
    conf = ConfigPrimary()
    hid_dim_lstm = conf.HIDDEN_DIM_LSTM
    loss_function = conf.LOSS_FUNCTION
    labels_num = conf.LABELS_NUM
    dropout = conf.DROPOUT
    lin_output_dim = conf.LINEAR_OUTPUT_DIM
    lr = conf.LEARNING_RATE
    epochs_num = conf.EPOCHS_NUM
    batch_size = conf.BATCH_SIZE
    momentum = conf.MOMENTUM
    threshold = conf.THRESHOLD
    return hid_dim_lstm, loss_function, labels_num, dropout, lin_output_dim, lr, epochs_num, batch_size, momentum, threshold


def create_models_df():
    models_dict = {'primary': {}}
    columns_df = ['hid_dim_lstm', 'dropout', 'lin_output_dim', 'lr', 'epochs_num', 'batch_size', 'momentum', 'accuracy']
    for model_name in models_dict.keys():
        models_dict[model_name] = ['' for col_num in range(len(columns_df))]
    models_df = pd.DataFrame.from_dict(models_dict, orient='index')
    models_df.columns = columns_df
    return models_df


def load_pkls_df_primary(models_path):
    models_df = pd.read_csv(models_path+'models_df.csv', index_col=0)
    hid_dim_lstm, loss_function, labels_num, dropout, lin_output_dim, lr, epochs_num, batch_size, momentum, threshold = read_config_primary()
    models_df.loc[models_df.index == 'primary', 'hid_dim_lstm'] = hid_dim_lstm
    models_df.loc[models_df.index == 'primary', 'labels_num'] = 2
    models_df.loc[models_df.index == 'primary', 'loss_function'] = loss_function
    models_df.loc[models_df.index == 'primary', 'dropout'] = dropout
    models_df.loc[models_df.index == 'primary', 'lin_output_dim'] = lin_output_dim
    models_df.loc[models_df.index == 'primary', 'lr'] = lr
    models_df.loc[models_df.index == 'primary', 'epochs_num'] = epochs_num
    models_df.loc[models_df.index == 'primary', 'batch_size'] = batch_size
    models_df.loc[models_df.index == 'primary', 'momentum'] = momentum
    models_df.loc[models_df.index == 'primary', 'accuracy'] = round(0.0, 3)
    return models_df


def main():
    conf = ConfigMain()
    language = conf.LANGUAGE
    df = pd.read_csv(conf.DATA_FRAME)
    df.drop([col for col in df.columns if "Unnamed" in col], axis=1, inplace=True)
    train_dataset = df[:int(conf.TRAIN_RATIO * conf.ROWS_NUM)]
    dev_dataset = df[int(conf.TRAIN_RATIO * conf.ROWS_NUM):int(conf.TRAIN_RATIO * conf.ROWS_NUM) + int(conf.DEV_RATIO*conf.ROWS_NUM)]
    # dev_dataset = train_dataset.copy()
    test_dataset = df[int(conf.TRAIN_RATIO * conf.ROWS_NUM) + int(conf.DEV_RATIO*conf.ROWS_NUM):conf.ROWS_NUM]
    # classification_target = pd.read_csv(conf.CLASSIFICTION_TARGET)
    classification_target = dev_dataset.copy()
    dfs_list = [df[:len(df)], classification_target]
    # dfs_list = [df[:conf.ROWS_NUM]]
    word_dict = get_vocab(dfs_list, conf.FEATURE_TEXT)
    models_path = conf.MODELS_PATH
    file_name = conf.CLASSIFICATION_NAME
    if conf.PRIMARY_ALREADY_TRAINED:
        ACRCNN_test.classify(classification_target, train_dataset, word_dict, models_path, conf.MODEL_NAME, conf.THRESHOLD, conf.PROBLEM, file_name, language)
    else:
        models_df = create_models_df()
        models_df = ACRCNN_train.train_primary_model(train_dataset, dev_dataset, word_dict, models_df, models_path, language)
        models_df.to_csv(models_path + 'models_df.csv')
        ACRCNN_test.run_test(train_dataset, dev_dataset, word_dict, models_path, models_df, language)
    print('Done')


if __name__ == '__main__':
    main()
