import torch
import wikidata
import model
import pickle

def load_model(model_path, vocab_size, embedding_dim, hidden_size, num_layers):
    # Initialize the model
    lstm_model = model.init_model(vocab_size, embedding_dim, hidden_size, num_layers)
    # Load the trained model parameters
    lstm_model.load_state_dict(torch.load(model_path))
    lstm_model.eval()  # Set the model to evaluation mode
    return lstm_model

def tokenize_input(input_text, token_to_index):
    # Tokenize the input text and convert to indices
    tokens = wikidata.tokenize_text(input_text, use_nltk=True)
    token_indices = [token_to_index.get(token, 0) for token in tokens]  # Unknown tokens as 0
    return token_indices

def predict_next_tokens(model, input_indices, num_predictions, vocab_size):
    model.eval()  # Ensure model is in evaluation mode
    predictions = []
    input_tensor = torch.tensor(input_indices).unsqueeze(0)  # Add batch dimension

    for _ in range(num_predictions):
        with torch.no_grad():
            output = model(input_tensor)
            last_token_logits = output[0, -1, :]
            predicted_token_id = torch.argmax(last_token_logits, dim=-1).item()
            predictions.append(predicted_token_id)
            input_tensor = torch.cat([input_tensor[0], torch.tensor([predicted_token_id])]).unsqueeze(0)

    return predictions

# Model Hyperparameters (should match training configuration)
vocab_size = 76538  # Replace with your actual vocab size
embedding_dim = 256
hidden_size = 256
num_layers = 2
model_path = 'trained_lstm_model.pth'

# Load token_to_index and index_to_token
with open('token_to_index.pkl', 'rb') as f:
    token_to_index = pickle.load(f)

with open('index_to_token.pkl', 'rb') as f:
    index_to_token = pickle.load(f)

# Load the trained model
lstm_model = load_model(model_path, vocab_size, embedding_dim, hidden_size, num_layers)

# Example input text
input_text = "The majority of material"
input_indices = tokenize_input(input_text, token_to_index)  # token_to_index from your training script

# Predict the next 5 tokens
num_predictions = 10
predicted_indices = predict_next_tokens(lstm_model, input_indices, num_predictions, vocab_size)

# Convert indices to tokens
predicted_tokens = [index_to_token[idx] for idx in predicted_indices]  # index_to_token from your training script

print("Predicted tokens:", predicted_tokens)
