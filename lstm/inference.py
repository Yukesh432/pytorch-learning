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

# def predict_next_tokens(model, input_indices, num_predictions, vocab_size):
#     model.eval()  # Ensure model is in evaluation mode
#     predictions = []
#     input_tensor = torch.tensor(input_indices).unsqueeze(0)  # Add batch dimension

#     for _ in range(num_predictions):
#         with torch.no_grad():
#             output = model(input_tensor)
#             last_token_logits = output[0, -1, :]
#             predicted_token_id = torch.argmax(last_token_logits, dim=-1).item()
#             predictions.append(predicted_token_id)
#             input_tensor = torch.cat([input_tensor[0], torch.tensor([predicted_token_id])]).unsqueeze(0)

#     return predictions
def generate_text(model, start_prompt, token_to_index, index_to_token, gen_length, seq_length):
    model.eval()
    tokens = tokenize_input(start_prompt, token_to_index)
    token_indices = [token_to_index.get(token, token_to_index['<UNK>']) for token in tokens]
    generated_text = [index_to_token[idx] for idx in token_indices]  # Convert to string tokens

    for _ in range(gen_length):
        input_tensor = torch.tensor([token_indices[-seq_length:]], dtype=torch.long)  # Use the last seq_length tokens
        with torch.no_grad():
            output = model(input_tensor)
        next_token_index = output[0, -1, :].argmax().item()
        next_token = index_to_token[next_token_index]
        generated_text.append(next_token)  # Append the string token
        token_indices.append(next_token_index)  # Append the token index for next input

    return ' '.join(generated_text)



# Model Hyperparameters (should match training configuration)
vocab_size = 76538  # Replace with your actual vocab size
# vocab_size= 36998
embedding_dim = 256
hidden_size = 256
num_layers = 2
seq_length=30
model_path = 'trained_lstm_model2.pth'

# Load token_to_index and index_to_token
with open('token_to_index.pkl', 'rb') as f:
    token_to_index = pickle.load(f)

with open('index_to_token.pkl', 'rb') as f:
    index_to_token = pickle.load(f)

# Load the trained model
lstm_model = load_model(model_path, vocab_size, embedding_dim, hidden_size, num_layers)

# Example input text
# input_text = "Senjō no Valkyria 3 : Unrecorded"
# input_text = "Kennedy is the president of: "
# input_text= "The game takes place"
# input_text= " As with previous Valkyira Chronicles"
input_text= "this"

gen_length=50
generated_text = generate_text(lstm_model, input_text, token_to_index, index_to_token, gen_length, seq_length)

print("Generated text:", generated_text)
# input_indices = tokenize_input(input_text, token_to_index)  # token_to_index from your training script

# Predict the next 5 tokens
# num_predictions = 10
# predicted_indices = predict_next_tokens(lstm_model, input_indices, num_predictions, vocab_size)

# Convert indices to tokens
# predicted_tokens = [index_to_token[idx] for idx in predicted_indices]  # index_to_token from your training script

# print("Predicted tokens:", predicted_tokens)
