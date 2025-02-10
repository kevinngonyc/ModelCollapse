from enum import Enum

class TrainingParams:
    latent_dims = 20
    num_epochs = 30
    batch_size = 256
    collapse_size = 2560
    capacity = 64
    learning_rate = 1e-3
    variational_beta = 1
    alpha = 0.1
    gamma=15
    nl_stdev = 1
    ng_stdev = 100

class Paths:
    model_dir = "vae/outputs/models/"
    tensor_dir = "vae/outputs/tensors/"
    image_dir = "vae/images/"

class Labels:
    mod_labels = {
        "input": "input noise",
        "no_noise": "without noise",
        "synthetic": "real data mix",
        "latent": "latent noise",
        "gradient": "gradient noise",
        "gan": "VAE-GAN"
    }
    param_prefixes = {
        "input": ", stdev=", 
        "synthetic": ", synthetic ratio="
    }