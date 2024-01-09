import torch
import wikidata
import model
import pickle
import math

# Function to load the model
def load_model(model_path, vocab_size, embedding_dim, hidden_size, num_layers, device):
    lstm_model = model.TokenLSTM(vocab_size, embedding_dim, hidden_size, num_layers)
    lstm_model.load_state_dict(torch.load(model_path, map_location=device))
    lstm_model.to(device)
    lstm_model.eval()
    return lstm_model

# Function to evaluate perplexity
def evaluate_perplexity(model, data_loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
            total_loss += loss.item() * inputs.size(0)
    perplexity = math.exp(total_loss / len(data_loader.dataset))
    return perplexity

# Load the token_to_index and index_to_token mappings
with open('token_to_index.pkl', 'rb') as f:
    token_to_index = pickle.load(f)

# Load and preprocess the evaluation data
eval_file_path = 'wiki.valid.raw'  # Replace with your evaluation file path
eval_text = wikidata.load_data(eval_file_path)
eval_tokens = wikidata.tokenize_text(eval_text, use_nltk=True)
eval_token_indices = wikidata.tokens_to_indices(eval_tokens, token_to_index)
seq_length = 30  # Should be same as used during training
eval_sequences = wikidata.create_sequences(eval_token_indices, seq_length)
eval_inputs, eval_targets = wikidata.prepare_training_data(eval_sequences, seq_length)
eval_input_tensors, eval_target_tensors = wikidata.to_tensors(eval_inputs, eval_targets)

# Set device for evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
vocab_size = len(token_to_index)  # Should match the vocab size used during training
embedding_dim = 256  # Should match the model's embedding dimension used during training
hidden_size = 256  # Should match the model's hidden size used during training
num_layers = 2    # Should match the model's number of layers used during training
model_path = 'trained_lstm_model.pth'  # Path to your trained model
lstm_model = load_model(model_path, vocab_size, embedding_dim, hidden_size, num_layers, device)

# Create DataLoader for evaluation data
eval_data_loader = model.create_data_loader(eval_input_tensors, eval_target_tensors, batch_size=64)

# Evaluate the model and print perplexity
perplexity = evaluate_perplexity(lstm_model, eval_data_loader, device)
print(f"Perplexity on evaluation data: {perplexity}")
