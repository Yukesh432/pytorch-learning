import math
import torch
import torch.nn as nn

"""
Equation of LSTM cells:

a. f_t= sigmoid(Uf*x_t+ Vf*h_(t-1)+ b_f)
b. C_t'= f_t* C_(t-1)
c. i_t= sigmoid(Ui*x_t+ Vi*h_(t-1) + b_i)
d. g_t= tanh(Ug*x_t+ Vg*h_(t-1)+ b_g)
e. C_t= (i_t * g_t)+ f_t* C_(t-1)
f. h_t= o_t * tanh(C_t) 
"""


class CustomLSTM(nn.Module):
    def __init__(self, input_size:int, hidden_size:int):
        super().__init__()
        self.input_size= input_size
        self.hidden_size= hidden_size

        # Weight intitialization for input gate(i_t)
        self.U_i= nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_i= nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_i= nn.Parameter(torch.Tensor(hidden_size))

        # Forget gate
        self.U_f= nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_f= nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_f= nn.Parameter(torch.Tensor(hidden_size))

        # C_t
        self.U_c= nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_c= nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_c= nn.Parameter(torch.Tensor(hidden_size))

        # o_t
        self.U_o= nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_o= nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_o= nn.Parameter(torch.Tensor(hidden_size))
        
        self.init_weights()

    
    def init_weights(self):
        stdv= 1.0/math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    # Feed forward operation
    def forward(self, x, init_states= None):
        """Here X.shape is in the form of (batch_size, sequence_size, input_size)"""
        batch_size, seq_size, _= x.size()
        hidden_seq= []

        if init_states is None:
            h_t, c_t= (torch.zeros(batch_size, self.hidden_size).to(x.device),
                       torch.zeros(batch_size, self.hidden_size).to(x.device))
        else:
            h_t, c_t= init_states
        
        for t in range(seq_size):
            x_t= x[:, t, :]

            i_t= torch.sigmoid(x_t @ self.U_i+ h_t@ self.V_i + self.b_i)
            f_t= torch.sigmoid(x_t @ self.U_f + h_t@ self.V_f + self.b_f)
            g_t= torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t= torch.sigmoid(x_t @ self.U_o + h_t@self.V_o + self.b_o)
            c_t= f_t * c_t + i_t *g_t
            h_t= o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        # reshape hidden sequence
        hidden_seq= torch.cat(hidden_seq, dim=0)
        hidden_seq= hidden_seq.transpose(0,1).contiguous()
        return hidden_seq, (h_t, c_t)
    
    


