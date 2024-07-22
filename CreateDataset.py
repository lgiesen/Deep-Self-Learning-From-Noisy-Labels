import pandas as pd
from sklearn.model_selection import train_test_split

from config import *

# Read the file and process each line
data = []
with open(dataset_noisy, 'r') as f:
    lines = f.readlines()
    for line in lines:
        path, label = line.strip().split()
        data.append((path, int(label)))

# Convert the data into a DataFrame
df = pd.DataFrame(data, columns=['image_path', 'label'])

# Split the dataset
train_size = 47570
val_size = 14313
test_size = 10526

train_data, temp_data = train_test_split(df, train_size=train_size, random_state=42, stratify=df['label'])
val_data, test_data = train_test_split(temp_data, train_size=val_size, test_size=test_size, random_state=42, stratify=temp_data['label'])

# Save the datasets
train_data.to_csv(dataset_train_path, sep=' ', index=False, header=False)
val_data.to_csv(dataset_val_path, sep=' ', index=False, header=False)
test_data.to_csv(dataset_test_path, sep=' ', index=False, header=False)
# train_data.to_pickle(dataset_train_path)
# val_data.to_pickle(dataset_val_path)
# test_data.to_pickle(dataset_test_path)
print("Dataset splits created and saved successfully.")
