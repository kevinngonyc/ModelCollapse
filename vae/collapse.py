import time
import argparse
import os
import copy
from utils import device, load_model, create_dataloader, combine_dataloader
from train import train_lstm, train_vae, train_vae_gan, vae_loss, train_dataset
from config import TrainingParams, Paths, Labels
import torch

def generate_latent(lstm, vqvae, codebook):
    indices = TrainingParams.K_vecs * torch.ones(1, 1).long().to(device)
    for i in range(49):
        _, token = lstm(indices)
        token = token.squeeze(0)
        idx = TrainingParams.K_vecs
        while idx >= TrainingParams.K_vecs:
            idx = torch.multinomial(token, 1)
        indices = torch.cat((indices, idx.unsqueeze(0)), dim=1)
    
    latent = torch.empty(0, TrainingParams.latent_dims).to(device)
    for i in indices[0,1:]:
        latent = torch.cat((latent, codebook(i).unsqueeze(0)))
    latent = latent.view(1, 7, 7, -1)
    latent = latent.permute(0, 3, 1, 2)
    return latent

def generate_latent_batch(lstm, vqvae, codebook, n):
    latent_batch = torch.tensor([]).to(device)
    for i in range(n):
        latent_batch = torch.cat((latent_batch, generate_latent(lstm, vqvae, codebook)))
    return latent_batch

def record_values(vae_init, vae, samples, loss, discrim=None):
    latent = torch.randn(TrainingParams.collapse_size, TrainingParams.latent_dims, device=device)
    
    # reconstruct images from the latent vectors
    img_recon = vae.decoder(latent).clone().detach().to(device)
    samples.append(img_recon.data[:100].cpu())

    x_recon, latent_mu, latent_logvar = vae_init(img_recon)
    loss.append(vae_loss(x_recon, img_recon, latent_mu, latent_logvar))

    targets = None
    if discrim is not None:
        y, _ = discrim(img_recon)
        targets = y.clone().detach().to(device)

    return img_recon, targets

def vq_record_values(vae_init, vae_start, vae, samples, loss, lstm, codebook=False):
    vae.eval()
    lstm.eval()
    vae_init.eval()
    vae_start.eval()
    
    if codebook:
        latent = generate_latent_batch(lstm, vae, vae_start.vq.codebook, TrainingParams.collapse_size)
    else:
        latent = generate_latent_batch(lstm, vae, vae.vq.codebook, TrainingParams.collapse_size)
    
    # reconstruct images from the latent vectors
    img_recon = vae.decoder(latent).clone().detach().to(device)
    samples.append(img_recon.data[:100].cpu())

    x_recon, latent_mu, latent_logvar = vae_init(img_recon)
    loss.append(vae_loss(x_recon, img_recon, latent_mu, latent_logvar))

    return img_recon

def collapse(vae_init, vae_start, noise_input=False, noise_stdev=1, syn_ratio=1.0, debug=False, discrim=None, multiclass=False, generations=TrainingParams.generations):
    vae = copy.deepcopy(vae_start).to(device)

    samples = []
    loss = []

    for i in range(generations):
        if debug:
            print(f"Generation {i}")

        img_recon, targets = record_values(vae_init, vae, samples, loss, discrim=discrim)

        if noise_input:
            img_recon = torch.add(img_recon, torch.randn(TrainingParams.collapse_size, 1, 28, 28, device=device) * noise_stdev)
            img_recon = torch.clamp(img_recon, min=0, max=1)

        if syn_ratio < 1.0 and syn_ratio > 0:
            train_dataloader = combine_dataloader(img_recon, train_dataset, TrainingParams.collapse_size, TrainingParams.batch_size, syn_ratio)
        elif multiclass:
            train_dataloader = create_dataloader(img_recon, TrainingParams.collapse_size, TrainingParams.batch_size, targets=targets)
        else:
            train_dataloader = create_dataloader(img_recon, TrainingParams.collapse_size, TrainingParams.batch_size)

        if discrim is not None:
            train_vae_gan(vae, discrim, train_dataloader, multiclass=multiclass)
        else:
            train_vae(vae, train_dataloader)
    
    record_values(vae_init, vae, samples, loss)
    return samples, loss

def vq_collapse(vae_init, vae_start, lstm, codebook=False, lstm_train=False, debug=False, generations=20):
    vae = copy.deepcopy(vae_start).to(device)
    lstm = copy.deepcopy(lstm).to(device)

    samples = []
    loss = []

    for i in range(generations):
        if debug:
            print(f"Generation {i}")

        img_recon = vq_record_values(vae_init, vae_start, vae, samples, loss, lstm, codebook=codebook)

        train_dataloader = create_dataloader(img_recon, TrainingParams.collapse_size, TrainingParams.batch_size)
    
        train_vae(vae, train_dataloader)
        if lstm_train:
            train_lstm(vae, lstm, train_dataloader)
    
    vq_record_values(vae_init, vae_start, vae, samples, loss, lstm, codebook=codebook)

    return samples, loss

def collapse_and_save(vae_init, vae, suffix, noise_input=False, noise_stdev=1, syn_ratio=1.0, discrim=None, lstm=None, codebook=False, lstm_train=False, generations=TrainingParams.generations):
    if generations != TrainingParams.generations:
        suffix += f"_{generations}gen"
    if lstm is not None:
        samples, loss = vq_collapse(vae_init, vae, lstm, codebook=codebook, lstm_train=lstm_train, debug=True, generations=50)
    else:
        samples, loss = collapse(vae_init, vae, noise_input, noise_stdev, syn_ratio, discrim=discrim, generations=generations)
    torch.save(samples, os.path.join(Paths.tensor_dir, f"samples_{suffix}.pt"))
    torch.save(loss, os.path.join(Paths.tensor_dir, f"loss_{suffix}.pt"))

if __name__=="__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=Labels.mode_labels.keys(), help="mode of collapse")
    param_modes = ", ".join(Labels.param_prefixes.keys())
    parser.add_argument("-p", "--parameter", type=float, help=f"parameter for {{{param_modes}}}")
    parser.add_argument("-g", "--generations", type=int, help=f"number of generations")
    args = parser.parse_args()

    vae_init = load_model(os.path.join(Paths.model_dir, "vae_init.pt"))

    if args.mode in ["input", "nonoise", "synthetic"]:
        vae = vae_init
    elif args.mode == "latent":
        vae = load_model(os.path.join(Paths.model_dir, "vae_latent.pt"))
    elif args.mode == "gradient":
        vae = load_model(os.path.join(Paths.model_dir, "vae_gradient.pt"))
    elif args.mode == "gan":
        vae = load_model(os.path.join(Paths.model_dir, "vae_gan.pt"))
        discrim = load_model(os.path.join(Paths.model_dir, "discrim.pt"))
    elif args.mode in ["vqvae", "vqvae_lstm", "vqvae_codebook", "vqvae_lstm_codebook"]:
        vae = load_model(os.path.join(Paths.model_dir, "vqvae.pt"))
        lstm = load_model(os.path.join(Paths.model_dir, "lstm.pt"))
    else:
        raise Exception("Need to specify valid noise type")

    generations = args.generations if args.generations is not None else TrainingParams.generations

    if args.mode == "input":
        noise_stdev = args.parameter if args.parameter is not None else 1
        suffix = f"{args.mode}_{noise_stdev}"
        collapse_and_save(vae_init, vae, suffix, noise_input=True, noise_stdev=noise_stdev, generations=generations)
    elif args.mode == "synthetic":
        syn_ratio = args.parameter if args.parameter is not None else 1.0
        suffix = f"{args.mode}_{syn_ratio}"
        collapse_and_save(vae_init, vae, suffix, syn_ratio=syn_ratio, generations=generations)
    elif args.mode == "gan":
        collapse_and_save(vae_init, vae, args.mode, discrim=discrim, generations=generations)
    elif args.mode == "vqvae":
        collapse_and_save(vae_init, vae, args.mode, lstm=lstm, generations=generations)
    elif args.mode == "vqvae_lstm":
        collapse_and_save(vae_init, vae, args.mode, lstm=lstm, lstm_train=True, generations=generations)
    elif args.mode == "vqvae_codebook":
        collapse_and_save(vae_init, vae, args.mode, lstm=lstm, codebook=True, generations=generations)
    elif args.mode == "vqvae_lstm_codebook":
        collapse_and_save(vae_init, vae, args.mode, lstm=lstm, lstm_train=True, codebook=True, generations=generations)
    else:
        collapse_and_save(vae_init, vae, args.mode, generations=generations)

    print(f"Runtime: {time.time() - start_time}")