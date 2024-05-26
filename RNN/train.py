# Ref: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from main import NUM_LETTERS
from main import load_data, line_to_tensor, random_training_example

class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size= hidden_size
        self.i2h= nn.Linear(input_size+ hidden_size, hidden_size)
        self.i2o= nn.Linear(input_size+ hidden_size, output_size)
        self.softmax= nn.LogSoftmax(dim=1) # 1,57.... we need "57"

    def forward(self, input_tensor, hidden_tensor):
        combined= torch.cat((input_tensor, hidden_tensor),1)
        hidden= self.i2h(combined)
        output= self.i2o(combined)
        output= self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    

def category_from_output(output):
    category_idx= torch.argmax(output).item()
    return all_categories[category_idx]


def train(line_tensor, category_tensor, rnn_model):
    hidden= rnn_model.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden= rnn_model(line_tensor[1], hidden)

    loss= criterion(output, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()


def predict(input_line):
    print(f"\n {input_line}")
    with torch.no_grad():
        line_tensor= line_to_tensor(input_line)
        hidden= rnn_model.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden= rnn_model(line_tensor[i], hidden)

        guess= category_from_output(output)
        print(guess)


if __name__=="__main__":
    category_lines, all_categories= load_data()
    num_categories= len(all_categories)

    n_hidden= 128
    rnn_model= RNN(NUM_LETTERS, n_hidden, num_categories)

    criterion= nn.NLLLoss()
    lr= 0.001
    optimizer= torch.optim.SGD(rnn_model.parameters(), lr= lr)

    current_loss= 0.0
    all_losses= []
    plot_steps, print_steps= 1000, 5000
    n_iters= 10000

    for i in tqdm(range(n_iters)):
        category, line, category_tensor, line_tensor= random_training_example(category_lines, all_categories)
        output, loss= train(line_tensor, category_tensor, rnn_model)
        current_loss+= loss

        if (i+1)% plot_steps == 0:
            all_losses.append(current_loss/plot_steps)
            current_loss= 0

        if (i+1)% print_steps == 0:
            guess= category_from_output(output)
            correct= "CORRECT" if guess == category else f"WRONG ({category})"
            print(f"{i+1} {(i+1)/n_iters*100} {loss:.4f} {line} / {guess} {correct}")

    plt.figure()
    plt.plot(all_losses)
    plt.show()


    while True:
        sentence = input("Input:")
        if sentence == "quit":
            break
        
        predict(sentence)