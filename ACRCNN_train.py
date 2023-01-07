import numpy as np
import torch
import torch.optim as optim
import random
import time
from model import RCNN
from torch.utils.data.dataloader import DataLoader
import main
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_to_file(model_name, text):
    print(text)
    fout = open(str(model_name) + ".txt", "a")
    fout.write(text + '\n')
    fout.close()


def print_summary(models_path, model_name, accuracy_list, accuracy_list_dev):
    write_to_file(models_path + model_name, "Accuracy Train:")
    write_to_file(models_path + model_name, str(accuracy_list))
    write_to_file(models_path + model_name, "Accuracy Dev:")
    write_to_file(models_path + model_name, str(accuracy_list_dev))
    write_to_file(models_path + model_name, "Best Accuracy Dev:")
    write_to_file(models_path + model_name, str(max(accuracy_list_dev)))
    write_to_file(models_path + model_name, "-------------------------------------------------------------")
    write_to_file(models_path + model_name, "")


def print_evaluate_def(acc_num, dev_len, loss_scalar, total_time):
    print("Dev accuracy for this epoch: %.3f" % float(acc_num / dev_len))
    print("loss for this epoch %.3f" % float(loss_scalar))
    print("total time: %.3f" % total_time)
    print()
    print()
    return


def evaluate_dev(model, dev_dataloader, dev_data_reader):
    start_dev = time.clock()
    print("Calculate Dev accuracy")
    acc_num = 0.0
    loss_scalar = 0.0
    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0
    dev_len = len(dev_data_reader.tweets)
    with torch.no_grad():
        for batch_idx, input_data in enumerate(dev_dataloader):
            sentence = input_data[0].squeeze(0)
            label = input_data[3]
            loss, prediction, prediction_prob = model(sentence, label)
            loss_scalar += loss.item()
            correctness = prediction == label.item()
            acc_num += correctness
            loss_scalar /= dev_len
            # only for twitter classifier
            if label.item() == 1:
                if prediction == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if prediction == 1:
                    fn += 1
                else:
                    tn += 1
                    # until here
    end_dev = time.clock()
    total_time = end_dev - start_dev
    print_evaluate_def(acc_num, dev_len, loss_scalar, total_time)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    print('precision: ', round(prec, 3))
    print('recall: ', round(rec, 3))
    print('F1: ', round((2 * prec * rec) / (prec + rec), 3))
    print('##################')
    print()
    return float(acc_num / dev_len), float(loss_scalar)


def print_train_epoch(epoch, correct_num, train_len, loss_scalar, total_time):
    print("train accuracy after epoch " + str(epoch + 1) + " is: %.3f" % float(correct_num / train_len))
    print("loss after epoch ", epoch + 1, "is: %.3f" % float(loss_scalar))
    print("total time: %.3f" % total_time)
    print()
    return


def update_models_df(models_df, hid_dim_lstm, loss_function, dropout, lin_output_dim, lr, epochs_num, batch_size, momentum, best_dev_acc):
    models_df.loc[models_df.index == 'primary', 'hid_dim_lstm'] = hid_dim_lstm
    models_df.loc[models_df.index == 'primary', 'labels_num'] = 3
    models_df.loc[models_df.index == 'primary', 'loss_function'] = loss_function
    models_df.loc[models_df.index == 'primary', 'dropout'] = dropout
    models_df.loc[models_df.index == 'primary', 'lin_output_dim'] = lin_output_dim
    models_df.loc[models_df.index == 'primary', 'lr'] = lr
    models_df.loc[models_df.index == 'primary', 'epochs_num'] = epochs_num
    models_df.loc[models_df.index == 'primary', 'batch_size'] = batch_size
    models_df.loc[models_df.index == 'primary', 'momentum'] = momentum
    models_df.loc[models_df.index == 'primary', 'accuracy'] = round(best_dev_acc, 3)
    return models_df


def train_primary_model(train_dataset, dev_dataset, word_dict, models_df, models_path, language):
    hid_dim_lstm, loss_function, labels_num, dropout, lin_output_dim, lr, epochs_num, batch_size, momentum, threshold = main.read_config_primary()
    train_data_reader = main.preprocessing(train_dataset, word_dict, language)
    dev_data_reader = main.preprocessing(dev_dataset, word_dict, language)
    train_dataloader = DataLoader(train_data_reader, shuffle=False)
    dev_dataloader = DataLoader(dev_data_reader, shuffle=False)
    word_vectors = train_data_reader.word_vectors
    model = RCNN(word_vectors, hid_dim_lstm, loss_function, labels_num, dropout, lin_output_dim, batch_size)
    if language == 'heb':
        model = model.float()
    if torch.cuda.is_available():
        model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    train_len = len(train_data_reader.tweets)
    best_dev_acc = 0
    acc_train_list = []
    acc_dev_list = []
    for epoch in range(epochs_num):
        start_train = time.clock()
        correct_num = 0
        loss_scalar = 0
        iter_counter = 1
        for batch_idx, input_data in enumerate(train_dataloader):
            sentence = input_data[0].squeeze(0)
            label = input_data[3]
            loss, prediction, best_prob = model(sentence, label)
            loss = loss / batch_size
            loss.backward()
            if iter_counter % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
            iter_counter += 1
            loss_scalar += loss.item()
            correct_num += prediction == label.item()
        loss_scalar /= train_len
        acc_train_list.append(round(float(correct_num / train_len), 3))
        end_train = time.clock()
        total_time = end_train - start_train
        print_train_epoch(epoch, correct_num, train_len, loss_scalar, total_time)
        dev_acc, dev_loss = evaluate_dev(model, dev_dataloader, dev_data_reader)
        acc_dev_list.append(round(dev_acc, 3))
        if dev_acc > best_dev_acc:
            torch.save(model.state_dict(), models_path + 'primary.pkl')
            best_dev_acc = dev_acc
    models_df = update_models_df(models_df, hid_dim_lstm, loss_function, dropout, lin_output_dim, lr, epochs_num, batch_size, momentum, best_dev_acc)
    print_summary(models_path, "primary", acc_train_list, acc_dev_list)
    return models_df
