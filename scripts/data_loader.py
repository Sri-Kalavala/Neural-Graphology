import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

class SignatureDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx]['filename'])
        image = Image.open(img_name).convert('L')  # convert to grayscale
        label = self.labels_df.iloc[idx]['is_genuine']
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

# Define transforms - resize images and convert to tensor
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Paths
csv_file = "data/processed/labels.csv"
root_dir = "data/processed/signatures"

# Create dataset and dataloader
dataset = SignatureDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Test: check a batch shape
images, labels = next(iter(dataloader))
print(images.shape)  # should be [32, 1, 128, 128]
print(labels[:10])
