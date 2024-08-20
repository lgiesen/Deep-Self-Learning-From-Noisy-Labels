import pandas as pd
from sklearn.model_selection import train_test_split

from config import *

# Read the file and process each line
data = []
with open(dataset, 'r') as f:
    lines = f.readlines()
    for line in lines:
        path, label = line.strip().split()
        # the filepath needs to be changed because the images are extracted into the extracted_images directory
        path = path.replace("images", "extracted_images")
        data.append((path, int(label)))

# Convert the data into a DataFrame
df = pd.DataFrame(data, columns=['image_path', 'label'])

# Split the dataset
"""
The paper Deep Self-Learning From Noisy Labels by Han et al. (2019) 
does not set a specific test train split ratio. 
However, it mention a ratio that is used in another context. 
Thus, this ratio is used here too:
train_size = 47570
val_size = 14313
test_size = 10526
total = train_size + val_size + test_size
print(train_size/total, val_size/total, test_size/total)
0.6569625322818986 0.19766879807758703 0.14536866964051431
"""
train_ratio = 0.65
val_ratio = 0.20
test_ratio = 0.15

# First split to get train and temp (val + test)
train_data, temp_data = train_test_split(df, train_size=train_ratio, random_state=42, stratify=df['label'])

# Calculate relative proportions for val and test sets from the temp set
temp_val_ratio = val_ratio / (val_ratio + test_ratio)
temp_test_ratio = test_ratio / (val_ratio + test_ratio)
# Second split to get validation and test sets
val_data, test_data = train_test_split(temp_data, train_size=temp_val_ratio, random_state=42, stratify=temp_data['label'])

# Save the datasets
train_data.to_csv(dataset_train_path, sep=' ', index=False, header=False)
val_data.to_csv(dataset_val_path, sep=' ', index=False, header=False)
# test_data.to_csv(dataset_test_path, sep=' ', index=False, header=False)

# Update test dataset
data = []
with open(dataset_clean, 'r') as f:
    lines = f.readlines()
    for line in lines:
        path, label = line.strip().split()
        # the filepath needs to be changed because the images are extracted into the extracted_images directory
        path = path.replace("images", "extracted_images")
        data.append((path, int(label)))

# Convert the data into a DataFrame
df = pd.DataFrame(data, columns=['image_path', 'label'])

# Save the datasets
df.to_csv(dataset_test_path, sep=' ', index=False, header=False)

print("Dataset splits created and saved successfully.")

