import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import typing
from pathlib import Path
import re

class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_size, self.input_size)
        
    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht1, ct1)
        
    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, \
                            self.hidden_size, requires_grad=True).to(device),\
                torch.zeros(self.n_layers,\
                            batch_size, self.hidden_size, requires_grad=True).to(device))

def preprocess(text:str) -> str:
    return re.sub("[^абвгдеёжзийклмнопрстуфхцчшщэьыюяъ ,.!?]", "", text)

def evaluate(model, char_to_idx, idx_to_char, start_text='. ', prediction_len=100, temp=0.3):
    hidden = model.init_hidden()
    start_text = preprocess(start_text.lower())
    idx_input = [char_to_idx[char] for char in start_text]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = ""
    
    _, hidden = model(train, hidden)
    
    inp = train[-1].view(-1, 1, 1)
    
    for i in range(prediction_len):
        output, hidden = model(inp.to(device), hidden)
        output_logits = output.cpu().data.view(-1)
        p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()
        top_index = np.random.choice(len(char_to_idx), p=p_next)
        inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
        predicted_char = idx_to_char[top_index]
        predicted_text += predicted_char
    
    return predicted_text

with Path('char_to_idx.pickle').open("rb") as f:
    char_to_idx = pickle.load(f)
    
with Path('idx_to_char.pickle').open("rb") as f:
    idx_to_char = pickle.load(f)

model = RNN(input_size=len(idx_to_char), hidden_size=256, embedding_size=64, n_layers=4)
model.load_state_dict(torch.load('syrchello.pt'))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
