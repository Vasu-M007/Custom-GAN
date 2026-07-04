import torch
from torch.utils.data import DataLoader,dataloader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from mnist_dataloader import MNISTDataloader 
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MNISTDataloader("train_images","train_labels.csv")

noise = torch.randn(32,100,device=device)
noise_dim = 100

#generator takes noise as input and generates a image same as input dimension 784 for mnist digits

class DCGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(noise_dim,128*4*4)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(128,64,kernel_size=2,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,16,kernel_size=2,stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,kernel_size=2,stride=2,padding=2),
            nn.Sigmoid()      
        )

    def forward(self,x):
        x = self.fc(x)
        x = x.view(-1,128,4,4)
        x = self.model(x)
        return x
    
model = DCGenerator().to(device)
fake_images = model(noise)

fake_images_np = fake_images.detach().cpu().numpy()
batch_size = fake_images_np.shape[0]

# for i in range(batch_size):
#     plt.subplot(1,fake_images_np.shape[0],i+1)
#     plt.imshow(fake_images_np[i][0],cmap='gray')
#     plt.axis('off')
#     plt.title(f"fake : {i+1}")

# plt.show()

if __name__ == "__main__":
    DCGenerator()
   