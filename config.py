"""
This file comprises global variables, which are used by multiple files.
"""
dataset_root = '/Volumes/Festplatte/MATIML/data/clothing1M/'
dataset_img = f'{dataset_root}images/'
dataset_masks = f'{dataset_root}annotations/'
dataset_noisy = f'{dataset_masks}noisy_label_kv.txt'
dataset_train_path = f'{dataset_root}train_dataset.csv'
dataset_val_path = f'{dataset_root}val_dataset.csv'
dataset_test_path = f'{dataset_root}test_dataset.csv'
class_names = [line.strip() for line in open(f'{dataset_masks}category_names_eng.txt')]

batch_size=128
num_epochs=15
momentum=0.9
weight_decay=5e-3
# learning rate
lr=0.002
# decrease lr by 10 every 5 epochs
step_size=5
gamma=0.1