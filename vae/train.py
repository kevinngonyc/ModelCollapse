import time
from models import VariationalAutoencoder, NoisyLatentVariationalAutoencoder, Discriminator
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from utils import device
import torch
from config import TrainingParams, Paths, Labels
import argparse
import os
from torch.autograd import Variable


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


def train_vae_gan(vae, discrim, train_dataloader, debug=False):
    criterion=torch.nn.BCELoss().to(device)
    optim_E=torch.optim.RMSprop(vae.encoder.parameters(), lr=TrainingParams.learning_rate)
    optim_D=torch.optim.RMSprop(vae.decoder.parameters(), lr=TrainingParams.learning_rate)
    optim_Dis=torch.optim.RMSprop(discrim.parameters(), lr=TrainingParams.learning_rate*TrainingParams.alpha)

    for epoch in range(TrainingParams.num_epochs):
        if debug:
            print(f"Epoch: {epoch}")
        prior_loss_list,gan_loss_list,recon_loss_list=[],[],[]
        dis_real_list,dis_fake_list,dis_prior_list=[],[],[]
        for i, (data,_) in enumerate(train_dataloader, 0):
            bs=data.size()[0]
            
            ones_label=Variable(torch.ones(bs,1)).to(device)
            zeros_label=Variable(torch.zeros(bs,1)).to(device)
            zeros_label1=Variable(torch.zeros(64,1)).to(device)
            datav = Variable(data).to(device)
            rec_enc, mean, logvar = vae(datav)
            z_p = Variable(torch.randn(64,TrainingParams.latent_dims)).to(device)
            x_p_tilda = vae.decoder(z_p)
            
            output = discrim(datav)[0]
            errD_real = criterion(output, ones_label)
            dis_real_list.append(errD_real.item())
            output = discrim(rec_enc)[0]
            errD_rec_enc = criterion(output, zeros_label)
            dis_fake_list.append(errD_rec_enc.item())
            output = discrim(x_p_tilda)[0]
            errD_rec_noise = criterion(output, zeros_label1)
            dis_prior_list.append(errD_rec_noise.item())
            gan_loss = errD_real + errD_rec_enc + errD_rec_noise
            gan_loss_list.append(gan_loss.item())
            optim_Dis.zero_grad()
            gan_loss.backward(retain_graph=True)
            optim_Dis.step()
            
            
            output = discrim(datav)[0]
            errD_real = criterion(output, ones_label)
            output = discrim(rec_enc)[0]
            errD_rec_enc = criterion(output, zeros_label)
            output = discrim(x_p_tilda)[0]
            errD_rec_noise = criterion(output, zeros_label1)
            gan_loss = errD_real + errD_rec_enc + errD_rec_noise
            
            
            x_l_tilda = discrim(rec_enc)[1]
            x_l = discrim(datav)[1]
            rec_loss = ((x_l_tilda - x_l) ** 2).mean()
            err_dec = TrainingParams.gamma * rec_loss - gan_loss 
            recon_loss_list.append(rec_loss.item())
            optim_D.zero_grad()
            err_dec.backward(retain_graph=True)
            optim_D.step()
            
            rec_enc, mean, logvar = vae(datav)
            x_l_tilda = discrim(rec_enc)[1]
            x_l = discrim(datav)[1]
            rec_loss = ((x_l_tilda - x_l) ** 2).mean()
            prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
            prior_loss = (-0.5 * torch.sum(prior_loss))/torch.numel(mean.data)
            prior_loss_list.append(prior_loss.item())
            err_enc = prior_loss + 5*rec_loss
            
            optim_E.zero_grad()
            err_enc.backward(retain_graph=True)
            optim_E.step()

train_dataset = datasets.MNIST(root='./data', train = True, download = True, transform = ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=TrainingParams.batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='./data', train = False, download = True, transform = ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=TrainingParams.batch_size, shuffle=True)

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("type", choices=Labels.model_types, help="model type")
    args = parser.parse_args()

    if args.type == "init":
        vae_init = VariationalAutoencoder().to(device)
        train_vae(vae_init, train_dataloader)
        torch.save(vae_init, os.path.join(Paths.model_dir, "vae_init.pt"))
    elif args.type == "latent":
        vae_latent = NoisyLatentVariationalAutoencoder().to(device)
        train_vae(vae_latent, train_dataloader)
        torch.save(vae_latent, os.path.join(Paths.model_dir, "vae_latent.pt"))
    elif args.type == "gradient":
        vae_gradient = VariationalAutoencoder().to(device)
        train_vae(vae_gradient, train_dataloader, grad_noise=True, ng_stdev=TrainingParams.ng_stdev)
        torch.save(vae_gradient, os.path.join(Paths.model_dir, "vae_gradient.pt"))
    elif args.type == "gan":
        vae_gan = VariationalAutoencoder().to(device)
        discrim = Discriminator().to(device)
        train_vae_gan(vae_gan, discrim, train_dataloader)
        torch.save(vae_gan, os.path.join(Paths.model_dir, "vae_gan.pt"))
        torch.save(discrim, os.path.join(Paths.model_dir, "discrim.pt"))
    else:
        raise Exception("Need to specify valid train type")


    print(f"Train Runtime: {time.time() - start_time}")
