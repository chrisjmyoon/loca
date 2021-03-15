import torch
from torch import nn
from torch.nn import functional as F
from torch import nn
import pdb

# Standard implementation of Encoder in AE
class VAEEncoder(nn.Module):
    def __init__(self, len_z=512):
        super().__init__()
        self.len_z = len_z
        # Encoder conv layers
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=1,out_channels=16,kernel_size=3,padding=1), # padding=1 to keep dims same
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
        )

        # Encoder linear layer
        self.fc_mu = nn.Linear(in_features=1024, out_features=self.len_z)
        self.fc_var = nn.Linear(in_features=1024, out_features=self.len_z)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        var = self.fc_var(out)
        return mu, var
        
# Standard implementation of Decoder in AE
class VAEDecoder(nn.Module):
    def __init__(self, len_z=512):
        super().__init__()
        self.len_z = len_z
        # Decoder linear layer
        self.linear = nn.Linear(in_features=self.len_z, out_features=1024)

        # Encoder conv layers
        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=16,out_channels=16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=16,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.recon_layer = nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.linear(x)
        out = out.view(out.size(0), 16, 4, 4, 4)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.recon_layer(out)
        return out

class VAE(nn.Module):
    def __init__(self, len_z=512, device="cpu", beta=1):
        super().__init__()
        self.len_z = len_z
        self.device = device
        self.beta = beta # 1 is standard VAE

        self.encoder = VAEEncoder(self.len_z)
        self.decoder = VAEDecoder(self.len_z)
        

    def encode(self, input):
        mu, log_var = self.encoder(input)
        return [mu, log_var]

    def decode(self, z):
        result = self.decoder(z)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def kl_loss_function(self, mu, log_var, kld_weight):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = self.beta*kld_weight*kld_loss
        return kld_loss

    def sample(self,num_samples,current_device,**kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


