import time
import torch
from torch.utils.data.dataloader import DataLoader
import main
import pandas as pd
from model import RCNN
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_to_file(model_name, text):
    print(text)
    fout = open(str(model_name) + ".txt", "a")
    fout.write(text + '\n')
    fout.close()


def read_model_parameters(model_name, models_path):
    conf_df = pd.read_csv(models_path + 'models_df.csv', index_col=0)
    hid_dim_lstm = int(conf_df.loc[model_name].loc['hid_dim_lstm'])
    dropout = conf_df.loc[model_name].loc['dropout']
    lin_output_dim = int(conf_df.loc[model_name].loc['lin_output_dim'])
    lr = conf_df.loc[model_name].loc['lr']
    batch_size = int(conf_df.loc[model_name].loc['batch_size'])
    momentum = conf_df.loc[model_name].loc['momentum']
    loss_function = nn.NLLLoss()  # conf_df.loc[model_name].loc['loss_function']
    labels_num = int(conf_df.loc[model_name].loc['labels_num'])
    return hid_dim_lstm, loss_function, dropout, lin_output_dim, lr, batch_size, momentum, labels_num


def load_models_parameters(word_vectors, models_path):
    models = {'primary': dict()}
    for model_name in models.keys():
        hid_dim_lstm, loss_function, dropout, lin_output_dim, lr, batch_size, momentum, labels_num = read_model_parameters(model_name, models_path)
        rel_word_vectors = word_vectors
        model = RCNN(rel_word_vectors, hid_dim_lstm, loss_function, labels_num, dropout, lin_output_dim, batch_size)
        state_dict = torch.load(models_path + model_name + '.pkl', map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models[model_name]['model'] = model
        models[model_name]['acc_test'] = 0.0
        models[model_name]['rel_labels'] = {1, 2}
    return models


def print_evaluate_test(acc_num, test_len, loss_scalar, total_time):
    print("Test Accuracy: %.3f" % float(acc_num / test_len))
    print("Loss for this Epoch %.3f" % float(loss_scalar))
    print("Total Time: %.3f" % total_time)
    print()
    print()
    return


def print_summary_test(models_path, model_name, prec, rec, f1):
    write_to_file(models_path + model_name, "prec: " + str(prec))
    write_to_file(models_path + model_name, "rec: " + str(rec))
    write_to_file(models_path + model_name, "F1: " + str(f1))
    write_to_file(models_path + model_name, "-------------------------------------------------------------")
    write_to_file(models_path + model_name, "")


def evaluate_test(model, test_dataloader, test_data_reader, language):
    start_dev = time.clock()
    acc_num = 0.0
    loss_scalar = 0.0
    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0
    test_len = len(test_data_reader.tweets)
    if language == 'heb':
        model = model.float()
    with torch.no_grad():
        for batch_idx, input_data in enumerate(test_dataloader):
            sentence = input_data[0].squeeze(0)
            label = input_data[3]
            loss, prediction, prediction_prob = model(sentence, label)
            loss_scalar += loss.item()
            correctness = prediction == label.item()
            acc_num += correctness
            loss_scalar /= test_len
            if label.item() == 2:
                if prediction == 2:
                    tp += 1
                else:
                    fp += 1
            else:
                if prediction == 2:
                    fn += 1
                else:
                    tn += 1
                    # until here
    end_dev = time.clock()
    total_time = end_dev - start_dev
    print_evaluate_test(acc_num, test_len, loss_scalar, total_time)
    prec = round(tp / (tp + fp), 3)
    rec = round(tp / (tp + fn), 3)
    f1 = round((2 * prec * rec) / (prec + rec), 3)
    return prec, rec, f1


def run_test(train_dataset, test_dataset, word_dict, models_path, models_df, language):
    test_data_reader = main.preprocessing(test_dataset, word_dict, language)
    test_dataloader = DataLoader(test_data_reader, shuffle=False)
    train_data_reader = main.preprocessing(train_dataset, word_dict, language)
    word_vectors = train_data_reader.word_vectors
    start_test = time.clock()
    print("Calculate Test accuracy")
    test_len = len(test_data_reader.tweets)
    models = load_models_parameters(word_vectors, models_path)
    prec, rec, f1 = evaluate_test(models['primary']['model'], test_dataloader, test_data_reader, language)
    print_summary_test(models_path, 'primary', prec, rec, f1)
    end_test = time.clock()
    total_time = end_test - start_test
    print("total time: %.3f" % total_time)
    print()
    print()
    return


def run_classification_sentiment(test_df, model, test_dataloader, threshold):
    test_df['sentiment_pred'] = ''
    test_df['sentiment_prob'] = 0
    with torch.no_grad():
        for batch_idx, input_data in enumerate(test_dataloader):
            sentence = input_data[0].squeeze(0)
            label = input_data[3]
            loss, prediction, prediction_prob = model(sentence, label)
            if prediction == 1:
                if prediction_prob > threshold:
                    test_df.loc[batch_idx, 'sentiment_pred'] = 'positive'
                else:
                    test_df.loc[batch_idx, 'sentiment_pred'] = 'neutral'
                test_df.loc[batch_idx, 'sentiment_prob'] = prediction_prob
            else:
                if prediction_prob > threshold:
                    test_df.loc[batch_idx, 'sentiment_pred'] = 'negative'
                else:
                    test_df.loc[batch_idx, 'sentiment_pred'] = 'neutral'
                test_df.loc[batch_idx, 'sentiment_prob'] = prediction_prob
    return test_df


def run_classification_relevance(test_df, model, test_dataloader, threshold):
    test_df['relevance_pred'] = False
    test_df['relevance_prob'] = 0
    with torch.no_grad():
        for batch_idx, input_data in enumerate(test_dataloader):
            sentence = input_data[0].squeeze(0)
            label = input_data[3]
            loss, prediction, prediction_prob = model(sentence, label)
            if prediction == 1:
                if prediction_prob > threshold:
                    test_df.loc[batch_idx, 'relevance_pred'] = True
                test_df.loc[batch_idx, 'relevance_prob'] = prediction_prob
            else:
                if prediction_prob > threshold:
                    test_df.loc[batch_idx, 'relevance_pred'] = False
                test_df.loc[batch_idx, 'relevance_prob'] = prediction_prob
    return test_df


def run_classification_category(test_df, model, test_dataloader, threshold):
    test_df['category_pred'] = ''
    test_df['category_prob'] = 0
    with torch.no_grad():
        for batch_idx, input_data in enumerate(test_dataloader):
            sentence = input_data[0].squeeze(0)
            label = input_data[3]
            loss, prediction, prediction_prob = model(sentence, label)
            if prediction_prob > threshold:
                if prediction == 1:
                    test_df.loc[batch_idx, 'category_pred'] = 'Irrelevant'
                elif prediction == 2:
                    test_df.loc[batch_idx, 'category_pred'] = 'News'
                else:
                    test_df.loc[batch_idx, 'category_pred'] = 'General'

            else:
                test_df.loc[batch_idx, 'category_pred'] = 'Unknown'
            test_df.loc[batch_idx, 'category_prob'] = prediction_prob
    return test_df


def classify(test_df, train_dataset, word_dict, models_path, model_name, threshold, problem, file_name, language):
    test_data_reader = main.preprocessing(test_df, word_dict, language, 'classify')
    test_dataloader = DataLoader(test_data_reader, shuffle=False)
    train_data_reader = main.preprocessing(train_dataset, word_dict, language)
    word_vectors = train_data_reader.word_vectors
    hid_dim_lstm, loss_function, dropout, lin_output_dim, lr, batch_size, momentum, labels_num = read_model_parameters('primary', models_path)
    model = RCNN(word_vectors, hid_dim_lstm, loss_function, labels_num, dropout, lin_output_dim, batch_size)
    state_dict = torch.load(models_path + model_name + '.pkl', map_location=device)
    model.load_state_dict(state_dict)
    model = model.float()
    model.to(device)
    print("Classification- Begin")
    if problem == 'sentiment':
        test_dataset = run_classification_sentiment(test_df, model, test_dataloader, threshold)
    elif problem == 'relevance':
        test_dataset = run_classification_relevance(test_df, model, test_dataloader, threshold)
    elif problem == 'category':
        test_dataset = run_classification_category(test_df, model, test_dataloader, threshold)
    #test_dataset = test_dataset[test_dataset['prediction'] == 1]
    #test_dataset = test_dataset[test_dataset['relevance_pred'] == True]
    test_dataset.to_csv(models_path + '//results//' + file_name + '.csv')
    return
