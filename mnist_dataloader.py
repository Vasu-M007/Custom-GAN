import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

transforms_t = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor()])

class MNISTDataloader(Dataset):
    def __init__(self,image_dir,label_path, is_train = True):
        self.image_dir = image_dir
        self.label_path = pd.read_csv(label_path)
        self.transform = transforms_t 

    def __len__(self):
        return len(self.label_path)
    
    def __getitem__(self,idx):
        ind = self.label_path.iloc[idx,1]
        ind = f"{ind}"
        image_path = os.path.join(self.image_dir,ind)
        image_path = image_path + ".png"
        labels = self.label_path.iloc[idx,0]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return image,labels
    
dataset = MNISTDataloader("train_images","train_labels.csv")
example = iter(dataset)
sample = next(example)

if __name__ == "__main__":
    MNISTDataloader("train_images","train_labels.csv")

