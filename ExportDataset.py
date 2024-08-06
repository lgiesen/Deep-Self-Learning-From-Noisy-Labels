import os
import shutil

from config import dataset, dataset_root


def copy_files(source_file, destination_folder):
    with open(source_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        file_path = line.split(' ')[0]
        relative_path = file_path.replace('images/', '')
        dest_path = os.path.join(destination_folder, relative_path)
        
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Copy the file
        shutil.copy(os.path.join(dataset_root, file_path), dest_path)
        print(f"Copied {file_path} to {dest_path}")

if __name__ == "__main__":
    destination_folder = f'{dataset_root}GoogleDrive_upload/images/'
    copy_files(dataset, destination_folder)
