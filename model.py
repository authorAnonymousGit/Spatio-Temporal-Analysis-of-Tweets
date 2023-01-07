import torch
import torch.nn as nn
from torch.nn import functional as F


class RCNN(nn.Module):
    def __init__(self, word_embeddings, hidden_dim_lstm, loss_function, labels_num, dropout, linear_output_dim,
                 batch_size):
        super(RCNN, self).__init__()
        self.word_embeddings = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
        self.vocab_dim = word_embeddings.shape[0]
        self.embeddings_dim = word_embeddings.shape[1]
        self.hidden_dim_lstm = hidden_dim_lstm
        self.W2_output_dim = linear_output_dim
        self.batch_size = batch_size
        self.loss = loss_function
        self.labels_dim = labels_num
        self.dropout = dropout
        self.dropout_linear = nn.Dropout(p=self.dropout)
        self.tanh = nn.Tanh()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(input_size=self.embeddings_dim, hidden_size=self.hidden_dim_lstm, dropout=self.dropout, bidirectional=True)
        # The output dim pf linear W2 might be changed since they did not discuss it in the paper
        self.linear_W2 = nn.Linear(2*self.hidden_dim_lstm+self.embeddings_dim, self.W2_output_dim)
        nn.init.xavier_uniform_(self.linear_W2.weight)
        nn.init.zeros_(self.linear_W2.bias)
        self.linear_W4 = nn.Linear(self.W2_output_dim, self.labels_dim)
        nn.init.xavier_uniform_(self.linear_W4.weight)
        nn.init.zeros_(self.linear_W4.bias)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sentence, ground_truth):
        sentence = sentence.to(self.device)
        words_embeds = self.word_embeddings(sentence)
        # label = torch.tensor([int(ground_truth) - 1])
        label = torch.tensor([int(ground_truth)])
        lstm_out, _ = self.lstm(words_embeds.unsqueeze(0))
        lstm_out = lstm_out.squeeze(0)
        X_cat = torch.cat([lstm_out, words_embeds], 1).to(self.device)
        Y2 = self.tanh(self.linear_W2(X_cat))
        # Y2 = self.dropout_linear(Y2)
        Y2 = Y2.unsqueeze(0)
        Y2 = Y2.permute(0, 2, 1)
        Y3 = F.max_pool1d(Y2, Y2.shape[2]).squeeze(2)
        Y4 = self.linear_W4(Y3)
        lsm = self.logsoftmax(Y4)
        soft_max = self.softmax(Y4)
        prediction = torch.argmax(lsm)
        prediction_prob = soft_max[0][prediction.item()].item()
        loss_val = self.loss(lsm, label.to(self.device))
        return loss_val, prediction.item(), prediction_prob
