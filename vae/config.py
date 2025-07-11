class TrainingParams:
    latent_dims = 20
    num_epochs = 30
    batch_size = 256
    collapse_size = 2560
    capacity = 64
    learning_rate = 1e-3
    variational_beta = 1
    alpha = 0.1
    gamma = 15
    beta = 0.25
    nl_stdev = 1
    ng_stdev = 100
    generations = 20
    K_vecs = 128
    lstm_dims = 128

class Paths:
    model_dir = "vae/outputs/models/"
    tensor_dir = "vae/outputs/tensors/"
    image_dir = "vae/images/"

class Labels:
    mode_labels = {
        "input": "input noise",
        "nonoise": "without noise",
        "synthetic": "real data mix",
        "latent": "latent noise",
        "gradient": "gradient noise",
        "gan": "VAE-GAN",
        "vqvae": "vqvae",
        "vqvae_lstm": "vqvae, lstm",
        "vqvae_codebook": "vqvae w/ original codebook",
        "vqvae_lstm_codebook": "vqvae, lstm w/ original codebook"
    }
    param_prefixes = {
        "input": ", stdev=", 
        "synthetic": ", synthetic ratio="
    }
    model_types = ["init", "latent", "gradient", "gan", "vqvae"]
    viz_types = ["single", "loss_init"]