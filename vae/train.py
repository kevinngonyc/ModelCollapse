import time
from models import VariationalAutoencoder, NoisyLatentVariationalAutoencoder
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from utils import device
import torch
from config import TrainingParams, Paths
import sys
import os


def vae_loss(recon_x, x, mu, logvar):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + TrainingParams.variational_beta * kldivergence


def train_vae(vae, train_dataloader, grad_noise=False, ng_stdev=1, debug=False):
    vae = vae.to(device)

    if debug:
        num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
        print('Number of parameters: %d' % num_params)

    optimizer = torch.optim.Adam(params=vae.parameters(), lr=TrainingParams.learning_rate, weight_decay=1e-5)
    vae.train()
    train_loss_avg = []

    if debug:
        print('Training ...')
    for epoch in range(TrainingParams.num_epochs):
        train_loss_avg.append(0)
        num_batches = 0

        for image_batch, _ in train_dataloader:
            image_batch = image_batch.to(device)

            # vae reconstruction
            image_batch_recon, latent_mu, latent_logvar = vae(image_batch)

            # reconstruction error
            loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            if grad_noise:
                for p in vae.parameters():
                    p.grad = p.grad + torch.randn_like(p.grad) * ng_stdev

            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()

            train_loss_avg[-1] += loss.item()
            num_batches += 1

        train_loss_avg[-1] /= num_batches
        if debug:
            print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, TrainingParams.num_epochs, train_loss_avg[-1]))

train_dataset = datasets.MNIST(root='./data', train = True, download = True, transform = ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=TrainingParams.batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='./data', train = False, download = True, transform = ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=TrainingParams.batch_size, shuffle=True)

if __name__ == "__main__":
    start_time = time.time()

    if sys.argv[1] == "init":
        vae_init = VariationalAutoencoder().to(device)
        train_vae(vae_init, train_dataloader)
        torch.save(vae_init, os.path.join(Paths.model_dir, "vae_init.pt"))
    elif sys.argv[1] == "latent":
        vae_latent = NoisyLatentVariationalAutoencoder().to(device)
        train_vae(vae_latent, train_dataloader)
        torch.save(vae_latent, os.path.join(Paths.model_dir, "vae_latent.pt"))
    elif sys.argv[1] == "gradient":
        vae_gradient = VariationalAutoencoder().to(device)
        train_vae(vae_gradient, train_dataloader, grad_noise=True, ng_stdev=TrainingParams.ng_stdev)
        torch.save(vae_gradient,  os.path.join(Paths.model_dir, "vae_gradient.pt"))
    else:
        raise Exception("Need to specify valid train type")


    print(f"Train Runtime: {time.time() - start_time}")
