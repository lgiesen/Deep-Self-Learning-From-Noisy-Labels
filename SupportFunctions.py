"""General Purpose"""
def free_up_ram():
    # remove unused variables to free up ram
    import gc
    gc.collect()
    torch.cuda.empty_cache()

"""Testing"""
import matplotlib.pyplot as plt
import torch

from config import class_names


def visualize_image(img, label=None):
    """
    Displays an image with a label on top.
    
    Parameters:
    img (torch.Tensor): The image tensor with shape [3, H, W] where H is height and W is width.
    label (str): The label to be displayed on top of the image.
    """
    assert img.shape == torch.Size([3, 224, 224])
    # Check if the tensor needs normalization to [0, 1] range for visualization
    if img.min() < 0 or img.max() > 1:
        img = (img - img.min()) / (img.max() - img.min())

    # Convert the tensor to [H, W, C] format for Matplotlib
    img_np = img.permute(1, 2, 0).numpy()

    # Create a plot
    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    if label:
        plt.title(label, fontsize=16)
    plt.axis('off')  # Hide axes for a cleaner look
    plt.show()

#compare first image of sample_loader and sample_dataset
class_id=5
sample_dataset, sample_loader = sample_images(class_id)
visualize_image(next(iter(sample_loader))[0], class_id)
visualize_image(sample_dataset[0], class_id)
visualize_image(next(iter(train_loader))[0][0], class_id)
visualize_image(train_dataset[0][0], class_id)

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = outputs.max(1)
    return predicted.eq(labels).sum().item()

# Function to evaluate the model
def evaluate_model(loader, model, criterion, device):
    import torch
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

"""Training"""
def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, epochs_no_improve, checkpoint_path):
    checkpoint_path = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'epochs_no_improve': epochs_no_improve
    }, checkpoint_path)
    print(f'Checkpoint saved at epoch {epoch}')

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['best_val_loss'], checkpoint['epochs_no_improve']

"""Evaluation"""
def load_model(model_path, num_classes):
    from collections import Counter

    import numpy as np
    import pandas as pd
    from sklearn.utils.class_weight import compute_class_weight
    from torchvision import models

    from config import dataset
    from config import model_path_standard as model_path
    from config import num_classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    data = pd.read_csv(dataset.replace("../","/content/"), header=None, sep=' ', usecols=[1], names=['label'])
    labels = data['label'].values
    label_counts = Counter(labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model, criterion