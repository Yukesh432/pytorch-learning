import torch
import nltk

# Function to load data from a file
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function for tokenization
def tokenize_text(text, use_nltk=False):
    if use_nltk:
        nltk.download('punkt')
        return nltk.word_tokenize(text)
    else:
        return text.split()

# Function to build a vocabulary from tokens
def build_vocab(tokens):
    vocab = set(tokens)
    token_to_index = {token: idx for idx, token in enumerate(vocab)}
    index_to_token = {idx: token for token, idx in token_to_index.items()}
    return token_to_index, index_to_token

# Function to convert tokens to indices
def tokens_to_indices(tokens, token_to_index):
    return [token_to_index[token] for token in tokens]

# Function to create sequences for training
def create_sequences(token_indices, seq_length):
    sequences = [token_indices[i:i + seq_length] for i in range(len(token_indices) - seq_length)]
    return sequences

# Function to prepare training data
def prepare_training_data(sequences, seq_length):
    # Ensure that inputs and targets are of the same length
    inputs = [sequence[:-1] for sequence in sequences]  # Exclude the last token for input
    targets = [sequence[1:] for sequence in sequences]  # Exclude the first token for target
    return inputs, targets

# Function to convert data to PyTorch tensors
def to_tensors(inputs, targets):
    input_tensors = torch.tensor(inputs, dtype=torch.long)
    target_tensors = torch.tensor(targets, dtype=torch.long)
    return input_tensors, target_tensors
