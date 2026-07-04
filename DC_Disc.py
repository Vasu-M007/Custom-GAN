import torch
import torch.nn as nn
from mnist_dataloader import MNISTDataloader 
from DC_gen import DCGenerator  
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_generator = DCGenerator().to(device)
noise = torch.randn(32,100,device=device)

fake_images = model_generator(noise)
flattened_image_batch = fake_images.view(32,-1).to(device)
# print(flattened_image_batch.shape)
input_dim = 784

class DC_Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=2,stride=2,padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,64,stride=2,kernel_size=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=2,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,256,kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(2*2*256,1)
        )

    def forward(self,x):
        x = self.model(x)
        return x

    
if __name__ == "__main__":
     DC_Disc()
  


    


    

