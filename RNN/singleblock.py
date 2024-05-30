import io
import os
import unicodedata
import string
import torch
import random
import glob
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# Global variables
ALL_LETTERS = string.ascii_letters + ",.;"
NUM_LETTERS = len(ALL_LETTERS)

# Convert unicode string to ASCII
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS)

# Load data
def load_data():
    category_lines = {}
    all_categories = []

    def find_files(path):
        return glob.glob(path)
    
    def read_lines(filename):
        lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]
    
    for filename in find_files('../data/name/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    return category_lines, all_categories

# Convert letter to index
def letters_to_index(letter):
    return ALL_LETTERS.find(letter)

# Convert letter to tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, NUM_LETTERS)
    tensor[0][letters_to_index(letter)] = 1
    return tensor

# Convert line to tensor
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, NUM_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letters_to_index(letter)] = 1
    return tensor

# Get random training example
def random_training_example(category_lines, all_categories):
    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]
    
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor

# RNN model
class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = torch.tanh(self.i2h(combined))
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    
# Get category from output
def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]

# Training function
def train(line_tensor, category_tensor, rnn_model):
    hidden = rnn_model.init_hidden()
    rnn_model.zero_grad()  # Ensure gradients are zeroed for all parameters
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn_model(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

# Prediction function
def predict(input_line):
    print(f"\n {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        hidden = rnn_model.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn_model(line_tensor[i], hidden)

        guess = category_from_output(output)
        print(guess)

if __name__ == "__main__":
    # Load data
    category_lines, all_categories = load_data()
    num_categories = len(all_categories)

    # Model parameters
    n_hidden = 10
    rnn_model = RNN(NUM_LETTERS, n_hidden, num_categories)

    # Loss and optimizer
    criterion = nn.NLLLoss()
    lr = 0.001
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=lr)

    # Training loop
    current_loss = 0.0
    all_losses = []
    plot_steps, print_steps = 1000, 5000
    n_iters = 20000

    for i in tqdm(range(n_iters)):
        category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)
        output, loss = train(line_tensor, category_tensor, rnn_model)
        current_loss += loss

        if (i + 1) % plot_steps == 0:
            all_losses.append(current_loss / plot_steps)
            current_loss = 0

        if (i + 1) % print_steps == 0:
            guess = category_from_output(output)
            correct = "CORRECT" if guess == category else f"WRONG ({category})"
            print(f"{i + 1} {(i + 1) / n_iters * 100:.2f}% {loss:.4f} {line} / {guess} {correct}")

    # Plot loss
    plt.figure()
    plt.plot(all_losses)
    plt.show()
