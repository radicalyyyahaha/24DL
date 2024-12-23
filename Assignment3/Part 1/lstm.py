from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        
        self.Wix = nn.Linear(input_dim, hidden_dim)
        self.Wih = nn.Linear(hidden_dim, hidden_dim)
        self.bi = nn.Parameter(torch.zeros(hidden_dim))
        
        self.Wfx = nn.Linear(input_dim, hidden_dim)
        self.Wfh = nn.Linear(hidden_dim, hidden_dim)
        self.bf = nn.Parameter(torch.zeros(hidden_dim))
        
        self.Wgx = nn.Linear(input_dim, hidden_dim)
        self.Wgh = nn.Linear(hidden_dim, hidden_dim)
        self.bg = nn.Parameter(torch.zeros(hidden_dim))
        
        self.Wox = nn.Linear(input_dim, hidden_dim)
        self.Woh = nn.Linear(hidden_dim, hidden_dim)
        self.bo = nn.Parameter(torch.zeros(hidden_dim))
        
        self.Wph = nn.Linear(hidden_dim, output_dim)
        self.bp = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        batch_size = x.size(0)
        h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        outputs = []

        for t in range(self.seq_length):
            x_t = x[:, t, :]  

            i_t = torch.sigmoid(self.Wix(x_t) + self.Wih(h_t) + self.bi) 
            f_t = torch.sigmoid(self.Wfx(x_t) + self.Wfh(h_t) + self.bf)  
            g_t = torch.tanh(self.Wgx(x_t) + self.Wgh(h_t) + self.bg)    
            o_t = torch.sigmoid(self.Wox(x_t) + self.Woh(h_t) + self.bo)   

            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t

            p_t = self.Wph(h_t) + self.bp 
            y_t = F.softmax(p_t, dim=1)  

            outputs.append(y_t)

        return torch.stack(outputs, dim=1)

    # add more methods here if needed