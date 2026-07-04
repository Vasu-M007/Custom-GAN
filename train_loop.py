import torch
import torch.nn as nn
from mnist_dataloader import MNISTDataloader
from DC_Disc import DC_Disc
from DC_gen import DCGenerator
# from Discriminator import DISCModel
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.0002
num_epochs = 45
batch_size = 32

dataset = MNISTDataloader('train_images','train_labels.csv')
train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True)
num_steps = len(train_loader)

gen_model = DCGenerator().to(device)
disc_model = DC_Disc().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer_generator = torch.optim.Adam(gen_model.parameters(),lr = learning_rate)
optimizer_discriminator = torch.optim.Adam(disc_model.parameters(),lr=learning_rate)

total_generator_loss = []
total_discriminator_loss = []

for epoch in range(num_epochs):
    epoch_loss_disc = []
    epoch_loss_gen = []
    for i,(images,_) in enumerate(train_loader):

        images = images.to(device)

        noise = torch.randn(batch_size,100,device=device)
        fake_images = gen_model(noise)
        flattened_fakes = fake_images.view(-1,1,28,28).detach()
        real_images = images.view(-1,1,28,28)

        real_image_pred = disc_model(real_images)
        real_labels = torch.ones(images.size(0),1).to(device)
        loss_discriminator_real = criterion(real_image_pred,real_labels)

        fake_image_pred = disc_model(flattened_fakes)
        fake_labels = torch.zeros(images.size(0),1).to(device)
        loss_discriminator_fake = criterion(fake_image_pred,fake_labels)

        disc_loss = loss_discriminator_fake + loss_discriminator_real

        optimizer_discriminator.zero_grad()
        disc_loss.backward()
        epoch_loss_disc.append(disc_loss.item())
        optimizer_discriminator.step()

        noise = torch.randn(batch_size,100,device=device)
        fake_images = gen_model(noise)
        flattened_fakes = fake_images.view(-1,1,28,28)

        fake_img_pred_gen = disc_model(flattened_fakes)
        gen_label = torch.ones(images.size(0),1).to(device)

        loss_generator = criterion(fake_img_pred_gen,gen_label)
                                        
        optimizer_generator.zero_grad()
        loss_generator.backward()
        epoch_loss_gen.append(loss_generator.item())
        optimizer_generator.step()

        if i%150 == 0:
           print(f"Epoch : [{epoch+1}/{num_epochs}], Step : [{(i+1)}/{num_steps}], Discriminator Loss : [{disc_loss.item():.4f}], Generator Loss : [{loss_generator.item():.4f}]")

    avg_epoch_loss_gen = sum(epoch_loss_gen)/len(train_loader)
    total_generator_loss.append(avg_epoch_loss_gen)

    avg_epoch_loss_disc = sum(epoch_loss_disc)/len(train_loader)
    total_discriminator_loss.append(avg_epoch_loss_disc)

    with torch.no_grad():
        testing_noise = torch.randn(8,100,device=device)
        fake_img_test = gen_model(testing_noise).view(-1,1,28,28).cpu()

    real_batch = images[:8].view(-1,1,28,28).cpu()

    fig,axes = plt.subplots(2,8,figsize = (14,4))
    for i in range(8):
        axes[0, i].imshow(real_batch[i][0], cmap='gray')
        axes[0, i].axis("off")

        axes[1, i].imshow(fake_img_test[i][0], cmap='gray')
        axes[1, i].axis("off")

    plt.suptitle(f"Epoch {epoch+1}: Real (Top) vs Fake (Bottom)")
    plt.tight_layout()
    plt.show()

print("---TRAINING FINISHED---")

torch.save(gen_model.state_dict(),"DC_Generator_model.pth")
torch.save(disc_model.state_dict(),"DC_Discriminator.pth")
print("Models saved sucessfully")

plt.plot(total_generator_loss)
plt.xlabel("Epoch")
plt.ylabel("Generator Loss")
plt.title("Generator Loss vs Epoch")
plt.savefig("Generator_loss.png")
plt.show()

plt.plot(total_discriminator_loss)
plt.xlabel("Epoch")
plt.ylabel("Discriminator Loss")
plt.title("Discriminator Loss vs Epoch")
plt.savefig("Discriminator_loss.png")
plt.show()












        





