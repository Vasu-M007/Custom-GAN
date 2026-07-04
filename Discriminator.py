import torch
import torch.nn as nn
from mnist_dataloader import MNISTDataloader 
from Generator import GeneratorModel
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_generator = GeneratorModel().to(device)
noise = torch.randn(16,64,device=device)

fake_images = model_generator(noise)
flattened_image_batch = fake_images.view(16,-1)
input_dim = 784

class DISCModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,1),      
        )

    def forward(self,x):
        x = self.model(x)

        return x
    
if __name__ == "__main__":
    model = DISCModel()
    output = model(flattened_image_batch)


    


    

