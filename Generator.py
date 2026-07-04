import torch
from torch.utils.data import DataLoader,dataloader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from mnist_dataloader import MNISTDataloader 
import matplotlib.pyplot as plt

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MNISTDataloader("train_images","train_labels.csv")

noise = torch.randn(16,64,device=device)
noise_dim = 64

#generator takes noise as input and generates a image same as input dimension 784 for mnist digits

class GeneratorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,784),
            nn.Sigmoid(),
        )

    def forward(self,x):
        x = self.model(x)
        x = x.view(-1,1,28,28)
        return x
    
model = GeneratorModel().to(device)
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
    GeneratorModel()
   










        

