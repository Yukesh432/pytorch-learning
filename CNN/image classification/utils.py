import torch
from tqdm import tqdm
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        # Set the model to training mode
        model.train()

        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Evaluate on the validation set
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()

        # Print epoch results
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {result['train_loss']:.4f}, Validation Loss: {result['val_loss']:.4f}, Validation Accuracy: {result['val_acc']:.4f}")

        history.append(result)
    
    return history

