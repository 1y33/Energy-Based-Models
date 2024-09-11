import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets
import torchvision.transforms as transform
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class EnergyModel(nn.Module):
    def __init__(self,mlp,mlp_multiplier):
        super().__init__()

        self.model = nn.Sequential(
            self.conv_layer(1,64), # 32 -> 16
            self.conv_layer(64,128), # 16 -> 8
            self.conv_layer(128,256), # 8 -> 4
            self.conv_layer(256,512), # 4 -> 2
        )
        self.final_size = 512 * 2 * 2
        if mlp:
            self.final_layer = self.mlp_layer(input=self.final_size,mlp_middle_size=mlp_multiplier)
        else:
            self.final_layer = nn.Linear(self.final_size,1)
    def conv_layer(self,input,output):
        return nn.Sequential(
            nn.Conv2d(input,output,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(output)
        )

    def mlp_layer(self,input,mlp_middle_size):
        return nn.Sequential(
            nn.Linear(input,mlp_middle_size),
            nn.ReLU(),
            nn.Linear(mlp_middle_size,1)
        )

    def forward(self,x):
        x = self.model(x)
        x = x.view(x.size(0),-1)
        x = self.final_layer(x)
        return x

def langevin_dynamics(model,x,steps=10,step_size=0.01):
    x = x.clone()
    x = x.detach()
    x = x.requires_grad_(True)
    for i in range(steps):
        energy = model(x).sum()
        energy.backward()
        with torch.inference_mode():
            x -= step_size * x.grad
        x.grad.zero_()
    return x.detach()

def train_ebm(model,dataloader,optimizer,device,n_sampels=100,epochs=10,noise_std=0.5,langevin_steps=10,langevin_step_size=0.01):
    len_dl = len(dataloader)
    reg_factor = 0.001
    model.train()
    for epoch in tqdm(range(epochs)):
        for i,(real_images,_) in enumerate(dataloader):
        #   print(f"{i}/{len_dl}")
            if i > n_sampels:
                break
            optimizer.zero_grad()
            real_images = real_images.to(device)

            noise = torch.rand_like(real_images) * noise_std
            fake_images = langevin_dynamics(model,noise,step_size=langevin_step_size,steps=langevin_steps)

            real_energy = model(real_images).mean()
            fake_energy = model(fake_images).mean()

            loss = real_energy - fake_energy
            loss = loss  + reg_factor * (real_energy **2 + fake_energy**2)
            loss.backward()

            optimizer.step()

        print(f"epoch: {epoch+1} ,loss: {loss.item()} ")
        sample_images(model,langevin_steps,langevin_step_size,device)

    return model

def sample_images(model,step,step_size,device):
    noise = torch.randn((16,1,32,32)).to(device)
    print(noise.shape)
    sample = langevin_dynamics(model,noise,step,step_size)
    sample = sample.cpu().permute(0,2,3,1) * 0.5 + 0.5
    sample = sample.detach().numpy()

    plt.figure(figsize=(8,8))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(sample[i],cmap="gray")
        plt.axis("off")
    plt.show()

def create_dataloader(batch_size=64):
    data_transform = transform.Compose([
        transform.Resize((32,32)),
        transform.ToTensor(),
    ])
    dataset = torchvision.datasets.MNIST(root="data",download=True,transform=data_transform)
    dataloader = DataLoader(dataset,batch_size=batch_size)

    return dataloader


def main():
    model = EnergyModel(mlp=None,mlp_multiplier=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    dataloader = create_dataloader(16)
    model = train_ebm(model,dataloader,optimizer, device,epochs=10)

main()