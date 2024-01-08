import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math
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

# # Training Function
# def train_model(model, data_loader, learning_rate, num_epochs):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#     for epoch in range(num_epochs):
#         for inputs, targets in data_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
#             loss.backward()
#             optimizer.step()

#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
def train_model(model, data_loader, learning_rate, num_epochs, device):
    print("Entering train_model function....")
    model.to(device)  # Move the model to the specified device (GPU/CPU)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to device

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:  # Print progress every 100 batches
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item()}')

        average_loss = total_loss / len(data_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss}')

def evaluate_model(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_count = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
            total_loss += loss.item() * inputs.size(0)
            total_count += inputs.size(0)

    average_loss = total_loss / total_count
    perplexity = math.exp(average_loss)
    return perplexity