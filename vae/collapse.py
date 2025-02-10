import time
import sys
import os
import copy
from utils import device, load_model, create_dataloader, combine_dataloader
from train import train_vae, train_vae_gan, vae_loss, train_dataset
from config import TrainingParams, Paths
import torch

def record_values(vae_init, vae, samples, loss):
    latent = torch.randn(TrainingParams.collapse_size, TrainingParams.latent_dims, device=device)
    
    # reconstruct images from the latent vectors
    img_recon = vae.decoder(latent).clone().detach().to(device)
    samples.append(img_recon.data[:100].cpu())

    x_recon, latent_mu, latent_logvar = vae_init(img_recon)
    loss.append(vae_loss(x_recon, img_recon, latent_mu, latent_logvar))

    return img_recon

def collapse(vae_init, vae_start, samples, loss, noise_input=False, noise_stdev=1, syn_ratio=1.0, debug=False, discrim=None):
    vae = copy.deepcopy(vae_start).to(device)

    for i in range(20):
        if debug:
            print(f"Generation {i}")

        img_recon = record_values(vae_init, vae, samples, loss)

        if noise_input:
            img_recon = torch.add(img_recon, torch.randn(TrainingParams.collapse_size, 1, 28, 28, device=device) * noise_stdev)
            img_recon = torch.clamp(img_recon, min=0, max=1)

        if syn_ratio < 1.0 and syn_ratio > 0:
            train_dataloader = combine_dataloader(img_recon, train_dataset, TrainingParams.collapse_size, TrainingParams.batch_size, syn_ratio)
        else:
            train_dataloader = create_dataloader(img_recon, TrainingParams.collapse_size, TrainingParams.batch_size)

        if discrim is not None:
            train_vae_gan(vae, discrim, train_dataloader)
        else:
            train_vae(vae, train_dataloader)
    
    record_values(vae_init, vae, samples, loss)

def collapse_and_save(vae_init, vae, suffix, noise_input=False, noise_stdev=1, syn_ratio=1.0, discrim=None):
    samples = []
    loss = []
    collapse(vae_init, vae, samples, loss, noise_input, noise_stdev, syn_ratio, discrim=discrim)
    torch.save(samples, os.path.join(Paths.tensor_dir, f"samples_{suffix}.pt"))
    torch.save(loss, os.path.join(Paths.tensor_dir, f"loss_{suffix}.pt"))

if __name__=="__main__":
    start_time = time.time()
    vae_init = load_model(os.path.join(Paths.model_dir, "vae_init.pt"))

    if sys.argv[1] in ["input", "no_noise", "synthetic"]:
        vae = vae_init
    elif sys.argv[1] == "latent":
        vae = load_model(os.path.join(Paths.model_dir, "vae_latent.pt"))
    elif sys.argv[1] == "gradient":
        vae = load_model(os.path.join(Paths.model_dir, "vae_gradient.pt"))
    elif sys.argv[1] == "gan":
        vae = load_model(os.path.join(Paths.model_dir, "vae_gan.pt"))
        discrim = load_model(os.path.join(Paths.model_dir, "discrim.pt"))
    else:
        raise Exception("Need to specify valid noise type")

    if sys.argv[1] == "input" and sys.argv[2].replace('.','',1).isdigit():
        noise_stdev = float(sys.argv[2])
        suffix = f"{sys.argv[1]}_{sys.argv[2]}"
        collapse_and_save(vae_init, vae, suffix, noise_input=True, noise_stdev=noise_stdev)
    elif sys.argv[1] == "synthetic" and sys.argv[2].replace('.','',1).isdigit():
        syn_ratio = float(sys.argv[2])
        suffix = f"{sys.argv[1]}_{sys.argv[2]}"
        collapse_and_save(vae_init, vae, suffix, syn_ratio=syn_ratio)
    elif sys.argv[1] == "gan":
        collapse_and_save(vae_init, vae, sys.argv[1], discrim=discrim)
    else:
        collapse_and_save(vae_init, vae, sys.argv[1])

    print(f"Runtime: {time.time() - start_time}")