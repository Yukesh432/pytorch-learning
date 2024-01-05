import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# LSTM Model Definition
class TokenLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super(TokenLSTM, self).__init__()
        self.vocab_size= vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, sequences):
        embedded = self.embedding(sequences)
        lstm_out, _ = self.lstm(embedded)
        out = self.linear(lstm_out)
        return out

# Function to Initialize the Model
def init_model(vocab_size, embedding_dim, hidden_size, num_layers=1):
    model = TokenLSTM(vocab_size, embedding_dim, hidden_size, num_layers)
    return model

# Function to Create DataLoader
def create_data_loader(input_tensors, target_tensors, batch_size=64):
    dataset = TensorDataset(input_tensors, target_tensors)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

# Training Function
def train_model(model, data_loader, learning_rate, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
