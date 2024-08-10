# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = outputs.max(1)
    return predicted.eq(labels).sum().item()

# Function to evaluate the model
def evaluate_model(loader, model, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # No need to calculate gradients during evaluation
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            correct_predictions += calculate_accuracy(outputs, labels)
            total_samples += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy