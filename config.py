"""
This file comprises global variables, which are used by multiple files.
"""
# load environment file
import os

# define env file path based on environment
if 'COLAB_GPU' in os.environ:
    # google colab
    env_path = '../drive/MyDrive/Colab_Notebooks/Deep_Self_Learning_From_Noisy_Labels/env'
    dataset_root = '../drive/MyDrive/Colab_Notebooks/Deep_Self_Learning_From_Noisy_Labels/'
    shared_folder_path = '/content/drive/MyDrive/Colab_Notebooks/Deep_Self_Learning_From_Noisy_Labels/images/'
else:
    # local
    from dotenv import load_dotenv
    env_path = '.env'
    load_dotenv()
    dataset_root = os.getenv('DATASETROOT')
    # the shared folder (shared_folder_path) is only possible in Google Colab as the data is shared there

checkpoint_path = f'{dataset_root}checkpoints/'.replace("../","/content/")

# set other filepaths
dataset_img = f'{dataset_root}extracted_images/'
dataset_masks = f'{dataset_root}annotations/'
dataset = f'{dataset_masks}noisy_label_kv.txt'
dataset_clean = f'{dataset_masks}clean_label_kv.txt'
dataset_train_path = f'{dataset_root}train_dataset.csv'
dataset_val_path = f'{dataset_root}val_dataset.csv'
dataset_test_path = f'{dataset_root}test_dataset.csv'
dataset_sample_path = f'{dataset_root}sample_dataset.csv'
writer_path_standard = dataset_root.replace("..", "/content") + 'runs/resnet50_standard'
writer_path_smp = dataset_root.replace("..", "/content") + 'runs/resnet50_smp'
model_path_standard = f'{dataset_root}models/standard.pth'.replace("../","/content/")
model_path_smp = f'{dataset_root}models/smp.pth'.replace("../","/content/")
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
randomly_sampled_img_count=320
num_prototypes=8
threshold_percentile=40