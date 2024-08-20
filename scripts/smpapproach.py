# -*- coding: utf-8 -*-
"""
## Load the Data
Mount Google Drive to access data and other repo files
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

"""Set the seed for reproducability"""

import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

"""Clone the repository to access the other relevant files"""

# Commented out IPython magic to ensure Python compatibility.
# clone the repo
!git clone https://github.com/lgiesen/Deep-Self-Learning-From-Noisy-Labels.git

# go to directory
# %cd Deep-Self-Learning-From-Noisy-Labels

"""## Data Preparation

Define the dataset
"""

# Commented out IPython magic to ensure Python compatibility.
from config import batch_size, dataset_test_path, dataset_train_path, dataset_val_path
from LoadDataset import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Define the transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # "These exact values are used for normalizing data that has been pre-trained
    # on the ImageNet dataset. They are based on the statistics of the ImageNet
    # dataset, which consists of a large number of natural images."
    # https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/

])

# Create datasets
train_dataset = CustomImageDataset(file_path=dataset_train_path, transform=transform)
val_dataset = CustomImageDataset(file_path=dataset_val_path, transform=transform)
test_dataset = CustomImageDataset(file_path=dataset_test_path, transform=transform)

# Create data loaders
# pinned memory can significantly speed up the transfer of data between the host and the device (GPU) because the GPU can directly access it
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %cd ../../

"""Extract the image files"""

"""
time to extract all 9 tar files:
CPU times: user 3min 46s, sys: 6min 13s, total: 10min
Wall time: 6min 6s
"""
import tarfile
import os
from config import shared_folder_path, dataset_img

# Function to extract and process files
def extract_and_process(tar_file_path, extract_to):
    with tarfile.open(tar_file_path, 'r') as tar_ref:
        tar_ref.extractall(extract_to)
        print(f"Extracted {tar_file_path} to {extract_to}")

parallel_extraction = True

from concurrent.futures import ThreadPoolExecutor

# Function to extract and process files
def extract_and_process(tar_file_path, extract_to):
    with tarfile.open(tar_file_path, 'r') as tar_ref:
        tar_ref.extractall(extract_to)
        print(f"Extracted {tar_file_path} to {extract_to}")

# Create the extraction directory if it doesn't exist
os.makedirs(dataset_img, exist_ok=True)

# List of tar files to extract
tar_files = [os.path.join(shared_folder_path, f"{i}.tar") for i in range(10)]

# Function to handle extraction in parallel
def extract_tar_files_parallel(tar_files, extract_to):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_and_process, tar_file, extract_to) for tar_file in tar_files if os.path.exists(tar_file)]
        for future in futures:
            try:
                future.result()  # Wait for the result to ensure any exceptions are raised
            except Exception as e:
                print(f"An error occurred: {e}")

# Extract tar files in parallel
extract_tar_files_parallel(tar_files, dataset_img)
print("The extracted tar files should result in the folders 0 to 9:")
!ls "{dataset_img}"

"""## Training

Define function to extract the dataset data to calculate class weights
"""

from config import randomly_sampled_img_count, dataset_sample_path, dataset
from sklearn.utils.class_weight import compute_class_weight

def get_data():
    # Calculate the balanced class weights because of an imbalanced dataset
    # Read the data again for higher efficiency
    data = pd.read_csv(dataset.replace("../","/content/"), header=None, sep=' ', names=['image', 'label'])
    # Convert the labels to a numpy array
    images = data['image'].values
    labels = data['label'].values
    return images, labels

# Calculate the balanced class weights because of an imbalanced dataset
_, labels = get_data()
# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

"""Define the model"""

from torchvision import models
from config import lr, momentum, weight_decay, gamma, step_size, num_classes, writer_path_smp as writer_path
import pandas as pd
import numpy as np

# check if the GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
# Initialize the model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# Modify the final fully connected layer to output 14 classes
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)
# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
# Initialize the learning rate scheduler: Decay LR by a factor of 0.1 every 5 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Define the hook function with the intermediate_output as the features
def hook(module, input, output):
    global intermediate_output
    intermediate_output = output.detach()

def reset_intermediate_output():
    global intermediate_output
    intermediate_output = None

# Register the hook to the layer before the FC layer (AdaptiveAvgPool2d)
hook = model.avgpool.register_forward_hook(hook)

# Parallelize training across multiple GPUs
model = torch.nn.DataParallel(model).to(device)

import time
from config import num_epochs, alpha, checkpoint_path
from torch.utils.tensorboard import SummaryWriter

# Initialize TensorBoard writer
writer = SummaryWriter(writer_path)

# Early stopping parameters
patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

"""Define functions to calculate the prototypes"""

def sample_images(class_id, num_samples=randomly_sampled_img_count):
    """
    This function randomly samples 1280 images for each of the 14 classes
    from the original noisy dataset (cf. Fig. 3 of "Deep Self-Learning From Noisy Labels").
    Output: DataLoader
    """
    # Split the dataset into images and labels
    images, labels = get_data()

    # Get indices of images corresponding to the current class
    class_indices = (labels == class_id).nonzero()[0]

    # Randomly sample the required number of images for this class
    sampled_indices = torch.randperm(len(class_indices))[:num_samples]

    # Collect the sampled images
    sampled_class_images = images[class_indices[sampled_indices]]

    # Convert the array to a list of strings with " -1" appended to each
    #sampled_class_images = [f"{item} -1" for item in sampled_class_images]
    sampled_class_images = np.char.replace(list(sampled_class_images), "images", "extracted_images")

    # Export the samples to CSV to create a loader just like for the datasets
    pd.DataFrame(sampled_class_images).to_csv(dataset_sample_path, sep=' ', index=False, header=False)
    sample_dataset = CustomImageDataset(file_path=dataset_sample_path, transform=transform, sampling=True)
    sample_loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return sample_loader

def cos_similarity(samples_features, randomly_sampled_img_count=randomly_sampled_img_count):
    """
    This function computes the cosine similarity matrix for 1280 sample features for one of the 14 classes
    Input: Sample features are the randomly sampled 1280 images from one class. It is a tensor([1280, 3, 224, 224])
    Output: Similarity matrix of 1280 sample features for one of the 14 classes
    tensor([1280, 1280])
    """
    # Step 1: Flatten the image features to shape [1280, 150528]
    flattened_features = samples_features.view(randomly_sampled_img_count, -1)

    # Step 2: Normalize the features
    normalized_features = torch.nn.functional.normalize(flattened_features, p=2, dim=1)

    # Step 3: Compute the cosine similarity matrix
    similarity_matrix = torch.mm(normalized_features, normalized_features.t())

    return similarity_matrix

from config import threshold_percentile
def calc_similarity_threshold(similarity_matrix, percentile=threshold_percentile):
    """
    This function calculates a suitable similarity threshold value for the calc_rho_density function.
    It takes the similarity value of the data point at the specified percentile rank.

    Input:
    - similarity_matrix: cosine similarity matrix of 1280 images, torch.Size([1280, 1280])
    - percentile: the desired percentile for threshold calculation (default is 40)

    Output:
    - threshold: numeric value between 0.0 and 1.0 representing the similarity threshold
    """
    # Flatten the matrix and exclude the diagonal elements
    flattened_matrix = similarity_matrix.flatten()
    n = similarity_matrix.size(0)
    mask = torch.eye(n, dtype=torch.bool)
    non_diagonal_elements = flattened_matrix[~mask.flatten()]

    # Sort the non-diagonal elements
    sorted_elements = torch.sort(non_diagonal_elements).values

    # Calculate the index for the desired percentile
    index = int((percentile / 100.0) * len(sorted_elements))

    # Retrieve the similarity threshold value
    return sorted_elements[index].item()

def calc_rho_densities(similarity_matrix, threshold):
    """
    This function computes the density of the images of one class to determine
    if they are diverse prototypes in the later stages of the programming.
    It follows the following function: sign(x) = 1 if x > 0; sign(x) = 0 if x = 0; otherwise sign(x) = -1
    So, if the cosine similarity (S_c) of the two images exceed the threshold, then the density is incremented by one.
    If the threshold is equal to the S_c, then the density stays the same.
    If the threshold is larger than S_c, then the density is reduced by 1.
    Input: cosine similarity matrix of 1280 images (torch.Size([1280, 1280]) and the threshold (numeric value between 0 and 1)
    Output: Densities of 1280 sample features for each of the 14 classes
    tensor([1280])
    """
    # Initialize the density vector with zeros
    densities = torch.zeros(similarity_matrix.size()[0])

    # Loop through each image
    for i in range(similarity_matrix.size(0)):
        # Apply the sign function
        sign_values = torch.sign(similarity_matrix[i] - threshold)

        # Calculate the density for the i-th image
        densities[i] = torch.sum(sign_values)
    return densities


def calc_eta_similarity_measurement(similarity_matrix, densities):
    """
    This function computes the eta value, which describes the similarity
    measurement used to identify diverse and representative prototypes.
    Input: cosine similarity matrix of 1280 images (torch.Size([1280, 1280]) and len(densities) = 1280
    Output: The similarity measurement eta for each image
    tensor([1280])
    """
    # Initialize the eta tensor
    eta = torch.zeros_like(densities)

    # Get the maximum density value
    max_density = densities.max()

    for i in range(len(densities)):
        if densities[i] < max_density:
            # Find the maximum similarity for points with higher density
            mask = densities > densities[i]
            eta[i] = similarity_matrix[i, mask].max()
        else:
            # If the point has the maximum density, find the minimum similarity
            eta[i] = similarity_matrix[i].min()

    return eta

from config import num_prototypes
def select_prototypes(similarity_measurement, samples_features, num_prototypes=num_prototypes):
    """
    This function selects the prototypes for a class.
    Input: similarity_measurement (torch.Size([randomly_sampled_img_count])),
      samples_features (torch.Size([randomly_sampled_img_count, 2048, 1, 1])) and num_prototypes int
    Output: class_prototypes tensor (torch.Size([num_prototypes]))
    """
    # Sort the similarity_measurement tensor in descending order and get the indices
    sorted_indices = torch.argsort(similarity_measurement, descending=True)

    class_prototypes_indices = sorted_indices[:num_prototypes]
    # task: select the prototypes from the samples_features based on the indices in class_prototypes_indices
    class_prototypes = samples_features[class_prototypes_indices]
    return class_prototypes

from config import class_names
def correct_labels(prototypes, train_features, labels, inputs=None):
    """
    This function corrects the labels by calculating the average similarity scores for each class.
    The label of the highest scoring class is selected as the label.
    Input: prototypes (list with length of 14 for each class with each instance containing 8 prototypes each with a shape of torch.Size([num_prototypes, 2048, 1, 1])).
        train_features (torch.Size([len(train_dataset), 2048, 1, 1]]))
    Output: pseudo_labels (Tensor having a the same length as labels)
    """
    pseudo_labels = []

    # Iterate over each instance in train_features
    for image in train_features:
        # Calculate similarity scores for each class
        similarity_scores = []
        for class_prototypes in prototypes:
            # Calculate average similarity score for the class
            avg_similarity = torch.mean(torch.cosine_similarity(image, class_prototypes, dim=0))
            similarity_scores.append(avg_similarity)

        # Select the label of the highest scoring class
        highest_score_index = torch.argmax(torch.tensor(similarity_scores))
        pseudo_label = labels[highest_score_index]
        pseudo_labels.append(pseudo_label)

    # Print the percentage of labels changed
    changed_labels = sum(pseudo_labels[i] != labels[i] for i in range(len(labels)))
    percentage_changed = (changed_labels / len(labels)) * 100
    print(f"Percentage of labels changed: {percentage_changed}%")

    # Visualize the changed labels
    for i in range(len(labels)):
        if pseudo_labels[i] != labels[i]:
            from scripts.supportfunctions import visualize_image
            visualize_image(inputs[i], label=f"{class_names[labels[i]]} → {class_names[pseudo_labels[i]]}")

    return pseudo_labels

from scripts.supportfunctions import evaluate_model, calculate_accuracy
# 1. Train the Model
for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    model.train() # set model to training mode
    epoch_start_time = time.time()  # Start time for the epoch

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in train_loader:
        # Move input and label tensors to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero out the optimizer
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # loss is computed differently in first epoch
        if epoch == 0:
            loss = criterion(outputs, labels)
        else:
            start_time = time.time()
            pseudo_labels = []
            for inputs, labels in train_loader:
                # reset intermediate_output
                reset_intermediate_output()
                model(inputs)
                # extract features of the samples with the hook
                # (automatically saved as intermediate_output)
                batch_pseudo_labels = correct_labels(prototypes,
                    intermediate_output, labels)
                pseudo_labels.extend(batch_pseudo_labels)
            # calculate the loss on pseudo and original labels
            loss_original = (1-alpha) * criterion(outputs, labels)
            loss_pseudo = alpha * criterion(outputs, pseudo_labels)
            loss = loss_original + loss_pseudo
            end_time = time.time()
            print(f"SMP Loss calculation time: {(end_time - start_time) / 60} minutes")
        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct_predictions += calculate_accuracy(outputs, labels)
        total_samples += labels.size(0)

    # Save checkpoint
    save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, epochs_no_improve, checkpoint_path)


    # 2. Label Correction Phase (except in last iteration)
    with torch.no_grad():
        if epoch != num_epochs-1:
            # 2.1 Prototype Selection
            # initialize prototypes
            prototypes = []
            for class_id in range(num_classes):
                # Reset the features
                reset_intermediate_output()
                # sample m=1280 images for the current class
                sample_loader = sample_images(class_id)
                # feed the randomly sampled images through the model to extract
                # the features of the samples with the hook
                # (automatically in the intermediate_output variable)
                samples_features = []
                for inputs in sample_loader:
                    inputs = inputs.to(device)
                    model(inputs)
                    samples_features.append(intermediate_output)
                    reset_intermediate_output()
                del inputs
                # convert to tensor for following operations
                samples_features = torch.cat(samples_features)
                # calculate similarity matrix of sample features
                similarity_matrix = cos_similarity(samples_features)
                # calculate the threshold
                threshold = calc_similarity_threshold(similarity_matrix)
                # calculate density of images
                densities = calc_rho_densities(similarity_matrix, threshold)
                # calculate similarity measurement
                similarity_measurement = calc_eta_similarity_measurement(similarity_matrix, densities)
                del similarity_matrix, densities
                # select prototypes for each class
                class_prototypes = select_prototypes(similarity_measurement, samples_features)
                del samples_features
                prototypes.append(class_prototypes)
            prototypes = torch.cat(prototypes)

    epoch_duration = time.time() - epoch_start_time  # End time for the epoch
    avg_loss = running_loss / len(train_loader)  # Average loss for the epoch
    accuracy = correct_predictions / total_samples  # Accuracy for the epoch

    # Log the training loss, accuracy, and duration to TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)
    writer.add_scalar('Time/train', epoch_duration, epoch)

    # Validate the model
    val_loss, val_accuracy = evaluate_model(val_loader, model, criterion, device)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    # Print the loss, accuracy, and time for every epoch
    print(f'Epoch {epoch+1}/{num_epochs}, '
           f'Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}, '
        #   f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, '
          f'Time: {epoch_duration:.2f} sec')

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            early_stop = True

    # Step the scheduler
    scheduler.step()


hook.remove()

"""
The training was manually, stopped with a KeyboardInterrupt 
because it became evident that the prototype quality is very low. 
Continuing to train a low performance model for a long time did 
not justiy the high GPU training costs incurred through Google 
Colab Pro+. Thus, the SMP approach training was not completed.
"""