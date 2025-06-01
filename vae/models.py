import torch
from torch import nn
import torch.nn.functional as F
from config import TrainingParams
from utils import device

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = TrainingParams.capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        self.fc_mu = nn.Linear(in_features=c*2*7*7, out_features=TrainingParams.latent_dims)
        self.fc_logvar = nn.Linear(in_features=c*2*7*7, out_features=TrainingParams.latent_dims)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.c = TrainingParams.capacity
        self.fc = nn.Linear(in_features=TrainingParams.latent_dims, out_features=self.c*2*7*7)
        self.conv2 = nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.c*2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar, sampleme=False):
        if self.training or sampleme:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

class NoisyLatentVariationalAutoencoder(nn.Module):
    def __init__(self):
        super(NoisyLatentVariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, nl_stdev=1):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        latent = torch.add(latent, torch.randn(TrainingParams.latent_dims, device=device) * nl_stdev)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar, sampleme=False):
        if self.training or sampleme:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        c = TrainingParams.capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        self.fc_z = nn.Linear(in_features=c*2*7*7, out_features=TrainingParams.latent_dims)
        self.fc_y = nn.Linear(in_features=TrainingParams.latent_dims, out_features=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        xl = x
        z = F.relu(self.fc_z(x))
        y = torch.sigmoid(self.fc_y(z))
        return y, xl
    
class VQEncoder(nn.Module):
    def __init__(self):
        super(VQEncoder, self).__init__()
        c = TrainingParams.capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=TrainingParams.latent_dims, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class VQDecoder(nn.Module):
    def __init__(self):
        super(VQDecoder, self).__init__()
        c = TrainingParams.capacity
        self.conv2 = nn.ConvTranspose2d(in_channels=TrainingParams.latent_dims, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x

class VQVariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VQVariationalAutoencoder, self).__init__()
        self.encoder = VQEncoder()
        self.decoder = VQDecoder()
        self.vq = VectorQuantizer()

    def forward(self, x):
        latent = self.encoder(x)
        vq_loss, encoding_indices, quantized = self.vq(latent)
        x_recon = self.decoder(quantized)
        return vq_loss, encoding_indices, x_recon

    def latent_sample(self, mu, logvar, sampleme=False):
        if self.training or sampleme:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

class VectorQuantizer(nn.Module):
    def __init__(self):
        super(VectorQuantizer, self).__init__()
        self.codebook = nn.Embedding(num_embeddings=TrainingParams.K_vecs, embedding_dim=TrainingParams.latent_dims)
        self.embedding_dim = TrainingParams.latent_dims

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        input_shape = x.shape

        flat_input = x.view(-1, 1, self.embedding_dim)

        distances = (flat_input - self.codebook.weight.unsqueeze(0)).pow(2).mean(2)

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        quantized = self.codebook(encoding_indices).view(input_shape)

        codebook_loss = F.mse_loss(quantized, x.detach())
        commitment_loss = F.mse_loss(quantized.detach(), x)
        loss = codebook_loss + TrainingParams.beta * commitment_loss

        if self.training:
            quantized = x + (quantized - x).detach()

        return loss, encoding_indices, quantized.permute(0, 3, 1, 2).contiguous()
    
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=TrainingParams.K_vecs+1, embedding_dim=TrainingParams.latent_dims)
        self.lstm = nn.LSTM(input_size=TrainingParams.latent_dims, hidden_size=TrainingParams.lstm_dims, batch_first=True)
        self.fc1 = nn.Linear(TrainingParams.lstm_dims, TrainingParams.lstm_dims // 4)
        self.fc2 = nn.Linear(TrainingParams.lstm_dims // 4, TrainingParams.K_vecs+1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.embedding(x)
        logits, _ = self.lstm(x)
        logits = F.relu(self.fc1(logits))
        logits = self.fc2(logits)
        logits = logits[:, -1, :]
        out = self.softmax(logits)
        return logits, out