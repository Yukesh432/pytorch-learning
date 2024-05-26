import torch
import numpy as np
from model import *
from load_data import *

# Assuming CharModel and other necessary imports are correctly defined above


def generate_text(model, start_prompt, char_to_int, int_to_char, n_vocab, generate_length=1000):
    model.eval()
    pattern = [char_to_int[c] for c in start_prompt.lower()]
    print('Prompt: "%s"' % start_prompt)
    with torch.no_grad():
        for i in range(generate_length):
            x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
            x = torch.tensor(x, dtype=torch.float32)
            prediction = model(x)
            index = prediction.argmax().item()
            print(int_to_char[index], end="")
            pattern.append(index)
            pattern = pattern[1:]
    print("\nDone.")


def load_checkpoint(model, checkpoint_path):
    """
    Loads a model checkpoint.
    """ 
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def inference_from_checkpoints(checkpoint_dir, char_to_int, int_to_char, n_vocab, start_prompt, generate_length=50):
    """
    Perform inference using multiple model checkpoints.
    """
    # List your checkpoint files here. Assuming they are named with a pattern like 'model_checkpoint_epoch_{epoch}.pth'
    # checkpoint_files = [f"{checkpoint_dir}_epoch_{epoch}.pth" for epoch in range(5, 11, 5)]  # Adjust range as needed
    checkpoint_files = [f"{checkpoint_dir}_epoch_{epoch}.pth" for epoch in range(5, 51, 5)]  # Adjust range as needed

    for checkpoint_path in checkpoint_files:
        model = CharModel(1, 256, 2, n_vocab, 0.2)  # Ensure this matches your model's architecture
        model = load_checkpoint(model, checkpoint_path)
        print(f"\nGenerating text with model from checkpoint: {checkpoint_path}")
        generate_text(model, start_prompt, char_to_int, int_to_char, n_vocab, generate_length)

def main():
    filename = "wonderland.txt"
    seq_length = 100
    dataX, dataY, char_to_int, n_vocab, _ = load_data(filename, seq_length)
    int_to_char = {i: c for c, i in char_to_int.items()}
    
    # start_prompt = "Alice noticed with "  # Example prompt
    start_prompt = "if she were looking over their shoulders"  # Example prompt
    checkpoint_dir = "model_checkpoint"  # Adjust as needed

    inference_from_checkpoints(checkpoint_dir, char_to_int, int_to_char, n_vocab, start_prompt, 50)  # Generate 100 characters

if __name__ == "__main__":
    main()
