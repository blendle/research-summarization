import torch
from torch import nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_layers=1, dropout_p=0.3):
        super(EncoderRNN, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout_p)

        self.h0 = nn.Parameter(torch.zeros(self.n_layers, 1, self.hidden_size))
        self.c0 = nn.Parameter(torch.zeros(self.n_layers, 1, self.hidden_size))

    def forward(self, sentence_embeddings):
        # Reshape into #sentences x batch (=1) x embedding_size
        embeddings = sentence_embeddings.view(sentence_embeddings.size(0), 1, -1)
        embeddings = self.dropout(embeddings)

        output, hidden = self.lstm(embeddings, (self.h0, self.c0))

        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout_p=0.3):
        super(DecoderRNN, self).__init__()

        # Keep parameters for reference
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Input size = embedding size + number of labels
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout_p)

    def forward(self, attention_output, sentence_embedding, last_hidden):
        # Reshape current sentence embedding into 1 x batch (=1) x embedding_size
        embedding = sentence_embedding.view(1, 1, -1)
        embedding = self.dropout(embedding)

        # Combine embedded input sentence, last, run through RNN
        rnn_input = torch.cat((embedding, attention_output.unsqueeze(0)), 2)
        output, hidden = self.lstm(rnn_input, last_hidden)

        # Return rnn output & hidden state
        return output, hidden


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Attention, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear1 = nn.Linear(self.input_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, decoder_output, encoder_output):
        # Prepare & concatenate inputs
        decoder_output = decoder_output.squeeze(0)
        x = torch.cat((decoder_output, encoder_output), 1)

        # Go through layers
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return F.softmax(x), F.log_softmax(x)