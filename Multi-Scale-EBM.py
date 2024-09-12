import torch
import torch.nn as nn
from torchvision import transforms as transform
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


# we will use mnist dataset resized to 32x32
# we can apply 4 convolutions and extract features

class conv_res_layer(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()

        self.conv_1 = nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(c_out)
        self.elu = nn.ELU(inplace=True)
        self.conv_2 = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        res = x
        x = self.conv_1(x)
        x = self.norm(x)
        x = self.elu(x)
        x = self.dropout(x)
        res = self.conv_2(res)
        x = x + res
        return x


class MLP(nn.Module):
    def __init__(self, f_in, f_out, factor):
        super().__init__()

        self.scale = int((f_in + f_out) / factor)
        self.linear_1 = nn.Linear(f_in, self.scale)
        self.norm = nn.BatchNorm1d(self.scale)
        self.linear_2 = nn.Linear(self.scale, f_out)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(x)
        x = self.norm(x)
        x = self.elu(x)
        x = self.linear_2(x)

        return x


class energy_net(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = conv_res_layer(1, 16)  # 32 -> 16
        self.layer_2 = conv_res_layer(16, 32)  # 16 -> 8
        self.layer_3 = conv_res_layer(32, 64)  # 8 -> 4
        self.layer_4 = conv_res_layer(64, 128)  # 4 -> 2

        self.layer_1_projection = MLP(16 * 16 * 16, 1, factor=8)
        self.layer_2_projection = MLP(32 * 8 * 8, 1, factor=4)
        self.layer_3_projection = MLP(64 * 4 * 4, 1, factor=2)
        self.layer_4_projection = MLP(128 * 2 * 2, 1, factor=1)

    def forward(self, x):
        x = self.layer_1(x)
        e_1 = self.layer_1_projection(x.view(x.shape[0], -1))

        x = self.layer_2(x)
        e_2 = self.layer_2_projection(x.view(x.shape[0], -1))

        x = self.layer_3(x)
        e_3 = self.layer_3_projection(x.view(x.shape[0], -1))

        x = self.layer_4(x)
        e_4 = self.layer_4_projection(x.view(x.shape[0], -1))

        return (e_1, e_2, e_3, e_4)


class EBM(nn.Module):
    def __init__(self):
        super().__init__()

        self.energy_net = energy_net()

        self.alphas = [0.25, 0.25, 0.25, 0.25]
        self.ld_steps = 20
        self.step_size = 0.01
        self.sigma = 0.3
        self.l2_lambda = 1e-4

    def get_energy(self, x):
        e1, e2, e3, e4 = self.energy_net(x)
        energy = e1 * self.alphas[0] + e2 * self.alphas[1] + e3 * self.alphas[2] + e4 * self.alphas[3]
        return energy

    def get_gen_loss(self, x, energy):
        x_sample = self.sample(x)
        fake_energy = self.get_energy(x_sample)
        loss = -(torch.logsumexp(energy, dim=1) - torch.logsumexp(fake_energy, dim=1))
        return loss

    def reconstruction_loss(self, x):
        x_sample = self.sample(x)
        loss = torch.nn.functional.mse_loss(x, x_sample)
        return loss

    def energy_gradient(self, x):
        self.energy_net.eval()
        x_i = x.data.to(device)
        x_i.requires_grad = True

        x_i_grad = torch.autograd.grad(torch.logsumexp(self.get_energy(x_i), dim=1).sum(), [x_i], retain_graph=True)[0]
        self.energy_net.train()
        return x_i_grad

    def langevine_dynamics_step(self, x_old, alpha):
        grad_energy = self.energy_gradient(x_old)
        epsilon = torch.randn_like(grad_energy) * self.sigma
        x_new = x_old + alpha * grad_energy + epsilon

        return x_new

    def sample(self, x):
        x_sample = 2. * torch.rand_like(x) - 1
        for i in range(self.ld_steps):
            x_sample = self.langevine_dynamics_step(x_sample, alpha=0.3)

        return x_sample

    def generate_images(self, n_images):
        noise = torch.randn((n_images, 1, 32, 32)).to(device)
        sample = self.sample(noise)
        sample = sample.cpu().permute(0, 2, 3, 1) * 0.5 + 0.5
        sample = sample.detach().numpy()

        plt.figure(figsize=(8, 8))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(sample[i], cmap="gray")
            plt.axis("off")
        plt.show()

    def forward(self, x):
        energy = self.get_energy(x)
        energy.unsqueeze(1)
        gen_loss = self.get_gen_loss(x, energy).mean()
        # rec_loss = self.reconstruction_loss(x)
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = gen_loss  # + rec_loss
        loss = loss + self.l2_lambda * l2_norm
        return loss


def create_dataloader(batch_size=64):
    data_transform = transform.Compose([
        transform.RandomHorizontalFlip(),
        transform.RandomVerticalFlip(),
        transform.Resize((32, 32)),
        transform.ToTensor(),
        transform.Normalize((0.5,), (0.5,))
    ])
    dataset = torchvision.datasets.MNIST(root="data", download=True, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2)

    return dataloader


def create_energies_plot(model, dataloader):
    model.eval()
    for x, _ in dataloader:
        x = x.to(device)
        energies = model.get_energy(x)
        break

    x = np.linspace(-1, 1, 100)
    energies = energies.cpu().detach().numpy()[:100]

    plt.plot(x, energies)
    plt.show()
    model.train()


def train_model(model, dataloader, n_epochs, sanity_check):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = []
    for epoch in tqdm(range(n_epochs)):
        for i, (x, y) in enumerate(dataloader):

            x = x.to(device)
            y = y.to(device)
            model.train()
            optimizer.zero_grad()
            loss = model(x)
            loss = loss.mean()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss)
            if sanity_check:
                break

        print(f"Epoch: {epoch} | Loss: {loss}")
        model.generate_images(16)
        create_energies_plot(model, dataloader)

    return model


def extract_conv_activations(model, input_tensor):
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output
        return hook

    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        model(input_tensor)

    for hook in hooks:
        hook.remove()

    return activations

def plot_conv_activation(activation, title="Activation"):
    if activation.ndimension() == 4:  # (batch_size, channels, height, width)
        num_channels = activation.size(1)
        fig, axes = plt.subplots(1, min(num_channels, 8), figsize=(15, 15))
        for i in range(min(num_channels, 8)):
            ax = axes[i] if num_channels > 8 else axes
            ax.imshow(activation[0, i].cpu().numpy(), cmap='gray')
            ax.axis('off')
        plt.suptitle(title)
        plt.show()
    else:
        print("Unsupported activation shape for convolutional visualization.")

def visualize_conv_activations(model, input_tensor):
    activations = extract_conv_activations(model, input_tensor)
    for name, activation in activations.items():
        print(f"Visualizing activation for: {name}")
        plot_conv_activation(activation, title=name)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = EBM().to(device)
dataloader = create_dataloader(256)
model = train_model(model,dataloader,10,sanity_check=False)

for x,y in dataloader:
  break
x.to("cpu")

visualize_conv_activations(model = model.energy_net.to("cpu"),input_tensor = x)


