"""
This file comprises global variables, which are used by multiple files.
"""
# load environment file
import os

# did not work somehow - to be checked later
# # define env file path based on environment
# if 'COLAB_GPU' in os.environ:
#     # google colab
#     env_path = '../drive/MyDrive/Colab_Notebooks/Deep_Self_Learning_From_Noisy_Labels/env'
# else:
#     # local
#     from dotenv import load_dotenv
#     env_path = '.env'
# load_dotenv()
# dataset_root = os.getenv('DATASETROOT')

if 'COLAB_GPU' in os.environ:
    # google colab
    dataset_root = '../drive/MyDrive/Colab_Notebooks/Deep_Self_Learning_From_Noisy_Labels/'
    shared_folder_path = '/content/drive/MyDrive/Colab_Notebooks/Deep_Self_Learning_From_Noisy_Labels/images/'
    checkpoint_path = f'{dataset_root}checkpoints/'.replace("../","/content/")
else:
    # local
    dataset_root = '/Volumes/Festplatte/MATIML/data/clothing1M/'


# set other filepaths
dataset_img = f'{dataset_root}extracted_images/'
dataset_masks = f'{dataset_root}annotations/'
dataset = f'{dataset_masks}noisy_label_kv.txt' # or clean: clean_label_kv
dataset_train_path = f'{dataset_root}train_dataset.csv'
dataset_val_path = f'{dataset_root}val_dataset.csv'
dataset_test_path = f'{dataset_root}test_dataset.csv'
# set dataset variables
class_names = [line.strip() for line in open(f'{dataset_masks}category_names_eng.txt')]
num_classes = len(class_names)
batch_size=128
# model variables
num_epochs=15
momentum=0.9
weight_decay=5e-3
# learning rate
lr=0.002
# decrease lr by 10 every 5 epochs
step_size=5
gamma=0.1
# weight factor alpha to balance the loss function for the original and corrected data
alpha=0.5
randomly_sampled_img_count=1280
num_prototypes=8
threshold_percentile=40