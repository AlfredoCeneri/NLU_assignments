import torch
import torch.nn as nn
from utils import DEVICE
from utils import WeightDrop

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, 
                 naive_dropout, n_layers = 1, pad_index = 0,
                 emb_dropout = 0.1, out_dropout = 0.1
                 ):

        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.lstm = nn.LSTM( emb_size, hidden_size, n_layers, batch_first = True)
         
        self.output = nn.Linear(hidden_size, output_size)
        
        self.naive_dropout = naive_dropout
        self.embedding_dropout = nn.Dropout(emb_dropout)
        self.output_dropout = nn.Dropout(out_dropout)
    
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)

        if self.naive_dropout and self.training:
            emb = self.embedding_dropout(emb)

        lstm_out, _ = self.lstm(emb)

        if self.naive_dropout and self.training:
            lstm_out = self.output_dropout(lstm_out)
            
        output = self.output(lstm_out).permute(0,2,1)
        return output 