import torch
import torch.nn as nn
from utils import DEVICE
from utils import WeightDrop

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, 
                 naive_dropout, variational_dropout=0, weight_tying=False,
                 n_layers = 1, pad_index = 0, emb_dropout = 0.1, out_dropout = 0.1
                 ):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.hidden_size = hidden_size
        self.layers = n_layers

        # variational dropout, drop hidden to hidden and input to hidden weights
        weight_names = [f'weight_hh_l{k}' for k in range(self.layers)]
        weight_names += [ f'weight_ih_l{k}' for k in range(self.layers)]

        module = nn.LSTM( emb_size, hidden_size, n_layers, batch_first = True)
        if variational_dropout != 0: 
            self.lstm = WeightDrop ( 
                module = module,
                weights = weight_names,
                dropout = variational_dropout,
                variational = True if variational_dropout != 0 else False
            )
        else:
            self.lstm = module
         
        self.output = nn.Linear(hidden_size, output_size)
        
        if weight_tying:
            if hidden_size == emb_size:             
                self.output.weight = nn.Parameter(self.embedding.weight)
            else:
                raise ValueError("shape mismatch: trying to tie weights of size {hidden_size} with weights of size {emb_size}")
            
        self.variational_dropout = variational_dropout
        self.naive_dropout = naive_dropout
        self.embedding_dropout = nn.Dropout(emb_dropout)
        self.output_dropout = nn.Dropout(out_dropout)
    
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)

        if self.variational_dropout and self.training:
            emb_mask = torch.zeros(1, 1, emb.size(2)).bernoulli_(1 - self.variational_dropout).to(DEVICE)
            emb_mask.expand(emb.size(0), emb.size(1), emb.size(2))
            emb = emb * emb_mask

        if self.naive_dropout and self.training:
            emb = self.embedding_dropout(emb)

        # applying variational dropout to lstm layers
        lstm_out, _ = self.lstm(emb)
         
        if self.variational_dropout and self.training:
            out_mask = torch.zeros(1, 1, lstm_out.size(2)).bernoulli_(1 - self.variational_dropout).to(DEVICE)
            out_mask.expand(lstm_out.size(0), lstm_out.size(1), lstm_out.size(2))
            lstm_out = lstm_out * out_mask

        if self.naive_dropout and self.training:
            lstm_out = self.output_dropout(lstm_out)
            
        output = self.output(lstm_out).permute(0,2,1)
        return output 