import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 2)  # 2 classes: genuine or forged
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128 -> 64
        x = self.pool(F.relu(self.conv2(x)))  # 64 -> 32
        x = x.view(-1, 32 * 32 * 32)          # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Test the model with a batch
model = SimpleCNN()
outputs = model(images)
print(outputs.shape)  # should be [32, 2]
