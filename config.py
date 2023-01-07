
import torch.nn as nn


class ConfigMain:
    DATA_FRAME = 'Cal_relevance_3690.csv'
    ROWS_NUM = 3690
    TRAIN_RATIO = 0.8
    DEV_RATIO = 0.2
    MODEL = 'RCNN'
    FEATURE_TEXT = 'body'
    FEATURE_TEXT_OVER_UNDER = 'body'
    MODEL_NAME = 'primary'
    LANGUAGE = 'eng'
    PROBLEM = 'relevance'
    MODELS_PATH = './/trained_models//' + PROBLEM + '_California//'
    PRIMARY_ALREADY_TRAINED = True
    THRESHOLD = 0.6
    CLASSIFICTION_TARGET = 'test_ISR_EQ.csv'
    CLASSIFICATION_NAME = 'validation_' + LANGUAGE

class ConfigPrimary:
    HIDDEN_DIM_LSTM = 110
    LOSS_FUNCTION = nn.NLLLoss()
    LABELS_NUM = 3
    DROPOUT = 0.0
    LINEAR_OUTPUT_DIM = 200
    LEARNING_RATE = 0.02
    EPOCHS_NUM = 7
    BATCH_SIZE = 56
    MOMENTUM = 0.7
    # This parameter is relevant only for the ACRCNN model
    THRESHOLD = 0.8
