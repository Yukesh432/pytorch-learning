import lstm.NTP.wikidata as wikidata, lstm.NTP.model as model
import torch


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Load the data
file_path = 'wiki.train.raw'
text_data = wikidata.load_data(file_path)

# Tokenize the text
tokens = wikidata.tokenize_text(text_data, use_nltk=True)

# Build the vocabulary
token_to_index, index_to_token = wikidata.build_vocab(tokens)

# Convert tokens to indices
token_indices = wikidata.tokens_to_indices(tokens, token_to_index)

# Define the sequence length
seq_length = 30

# Create sequences
sequences = wikidata.create_sequences(token_indices, seq_length)

# Prepare training data
inputs, targets = wikidata.prepare_training_data(sequences, seq_length)

# Convert to PyTorch tensors
input_tensors, target_tensors = wikidata.to_tensors(inputs, targets)

# Print tensor shapes for verification
print(input_tensors.shape)
print(target_tensors.shape)
print("......................................................................")

# Model Hyperparameters
vocab_size = len(token_to_index)  # Length of the vocabulary
embedding_dim = 256
hidden_size = 256
num_layers = 2
learning_rate = 0.001
num_epochs = 10
batch_size = 64

# Initialize the Model
lstm_model = model.init_model(vocab_size, embedding_dim, hidden_size, num_layers)

print('model defined.....................')

# Create DataLoader
data_loader = model.create_data_loader(input_tensors, target_tensors, batch_size)

print("Training Starts......................................")
# Train the Model
model.train_model(lstm_model, data_loader, learning_rate, num_epochs, device)
