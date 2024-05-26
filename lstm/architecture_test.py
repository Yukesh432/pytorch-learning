import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm

class CustomLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight initialization for input gate (i_t)
        self.U_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # Forget gate
        self.U_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # Cell state
        self.U_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        # Output gate
        self.U_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    # Feed forward operation
    def forward(self, x, init_states=None):
        """Here x.shape is in the form of (batch_size, sequence_length, input_size)"""
        batch_size, seq_size, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(x.device),
                        torch.zeros(batch_size, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        for t in range(seq_size):
            x_t = x[:, t, :]

            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
            g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        # Reshape hidden sequence
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

class LstmNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = CustomLSTM(100, 32)
        self.fc1 = nn.Linear(32, 2)

    def forward(self, x):
        x_, (h_n, c_n) = self.lstm(x)
        x_ = x_[:, -1, :]
        x_ = self.fc1(x_)
        return x_


# class CustomGRU(nn.Module):
#     def __init__(self, input_size:int, hidden_size: int):
#         super().__init__()
#         self.input_size= input_size
#         self.hidden_size= hidden_size

#         # input gate
#         self.U_i= nn.Parameter(torch.Tensor(input_size, hidden_size))
#         self.V_i= nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.b_i= nn.Parameter(torch.Tensor(hidden_size))

#         # reset gate
#         self.U_r= nn.Parameter(torch.Tensor())




# Initialize the model, loss function, and optimizer
model = LstmNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy data generation
batch_size = 16
sequence_length = 10
input_size = 100

# Create a batch of sequences, each of length 10 with 100-dimensional embeddings
dummy_data = torch.randn(batch_size, sequence_length, input_size)
print(dummy_data.shape)
dummy_labels = torch.randint(0, 2, (batch_size,))

# Training loop
num_epochs = 5
for epoch in tqdm(range(num_epochs)):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(dummy_data)
    
    # Calculate loss
    loss = criterion(outputs, dummy_labels)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
