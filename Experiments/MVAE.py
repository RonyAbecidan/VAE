import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils
from collections import OrderedDict
import matplotlib.pyplot as plt
import warnings
import pytorch_lightning as pl
import h5py as h5
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Build Encoder
class Encoder(pl.LightningModule):

    def __init__(self,latent_dim,im_size=32,in_channel=3,hiddens=[128,256,512]):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.im_size=im_size
        self.in_channel=in_channel
        self.hiddens=hiddens
        self.modules=[nn.Conv2d(self.in_channel,hiddens[0], kernel_size=3, stride=2, padding=1),  # hiddens[0]*im_size/2xim_size/2
                      nn.ReLU(),
                      nn.BatchNorm2d(hiddens[0])]

        for i in range(1,len(hiddens)):
            # hiddens[i]*(im_size/2^i)^2
            self.modules.append(nn.Conv2d(hiddens[i-1],hiddens[i], kernel_size=3, stride=2, padding=1))
            self.modules.append(nn.ReLU())
            self.modules.append(nn.BatchNorm2d(self.hiddens[i]))

        self.modules.append(nn.Tanh())
        self.modules.append(nn.Flatten())

        final_size=(im_size)//(2**(len(hiddens)))

        self.encode=nn.Sequential(*self.modules)

        self.fc_mu = nn.Linear(self.hiddens[-1] * final_size * final_size, self.latent_dim)
        self.fc_var = nn.Linear(self.hiddens[-1] * final_size * final_size, self.latent_dim)

    def forward(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encode(x)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]


# Build Decoder
class Decoder_MLP(pl.LightningModule):

    def __init__(self, latent_dim=4,in_channel=3,im_size=64,hiddens=[256,512]):
        super(Decoder_MLP, self).__init__()

        self.latent_dim = latent_dim
        self.im_size = im_size
        self.in_channel=in_channel
        self.hiddens=hiddens
        self.modules=[nn.Linear(latent_dim,hiddens[0])]

        for i in range(1,len(hiddens)):
            self.modules.append(nn.ReLU())
            self.modules.append(nn.Linear(hiddens[i-1],hiddens[i]))

        self.modules.append(nn.Tanh())
        self.modules.append(nn.Linear(hiddens[-1],self.in_channel*self.im_size*self.im_size))
        self.modules.append(nn.Sigmoid())

        self.decode=nn.ModuleList(self.modules)

    def forward(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        z=z.view(-1,self.latent_dim)
        for layer in self.decode:
            z = layer(z)
            
        
        z = z.view(-1,self.in_channel,self.im_size,self.im_size)

        return z

# Build Decoder
class Decoder_Conv(pl.LightningModule):

    def __init__(self, latent_dim=4, in_channel=3, im_size=64, hiddens=[512,256,128,64],init=4):
        super(Decoder_Conv, self).__init__()

        assert (2**(len(hiddens))*init==im_size),"Take care about the architecture of your decoder, there is something wrong"

        self.latent_dim = latent_dim
        self.im_size = im_size
        self.hiddens=hiddens
        self.in_channel = in_channel
        self.modules = [nn.ConvTranspose2d(self.latent_dim,hiddens[0],init, 1, 0), #init*init
                       nn.ReLU(),
                       nn.BatchNorm2d(hiddens[0])]

        for i in range(1, len(hiddens)):
            self.modules.append(nn.ConvTranspose2d(hiddens[i-1],hiddens[i], 4, 2, 1)) #im_size=2^(2+i)
            self.modules.append(nn.ReLU())
            self.modules.append(nn.BatchNorm2d(hiddens[i]))


        self.modules.append(nn.ConvTranspose2d(hiddens[-1],self.in_channel, 4, 2, 1)) #im_size*im_size
        self.modules.append(nn.Sigmoid())

        self.decode = nn.ModuleList(self.modules)

    def forward(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        z = z.view(-1,self.latent_dim,1,1)
        for layer in self.decode:
            z = layer(z)
           
        return z


# Build Decoder
class Decoder_Linear_Conv(pl.LightningModule):

    def __init__(self, latent_dim=4, in_channel=3, im_size=64, hiddens=[512,256,128,64],init=4):
        super(Decoder_Linear_Conv, self).__init__()

        assert (2**(len(hiddens))*init==im_size),"Take care about the architecture of your decoder, there is something wrong"

        self.latent_dim = latent_dim
        self.im_size = im_size
        self.hiddens= hiddens
        self.in_channel = in_channel
        self.modules = [nn.Linear(self.latent_dim,self.latent_dim//2),
                        nn.ReLU(),
                        nn.ConvTranspose2d(self.latent_dim//2,hiddens[0],init, 1, 0), #init*init
                        nn.SELU()]

        for i in range(1, len(hiddens)):
            self.modules.append(nn.ConvTranspose2d(hiddens[i-1],hiddens[i], 4, 2, 1)) #im_size=2^(2+i)
            self.modules.append(nn.SELU())

        self.modules.append(nn.ConvTranspose2d(hiddens[-1],self.in_channel, 4, 2, 1)) #im_size*im_size
        self.modules.append(nn.Sigmoid())

        self.decode = nn.ModuleList(self.modules)

    def forward(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        z = z.view(-1,self.latent_dim)
        for i,layer in enumerate(self.decode):
            if i==2:
                z=z.view(-1,self.latent_dim//2,1,1)
                
            z=layer(z)

        return z


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 1)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0, 1)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0, 1)
        nn.init.constant_(m.bias.data, 0)

# Decoders=nn.ModuleList([Decoder_MLP(latent_dim=100, in_channel=1, im_size=32, hiddens=[256,512]),
#           Decoder_Conv(latent_dim=100, in_channel=1, im_size=32, hiddens=[512,256,128],init=4),
#           Decoder_Linear_Conv(latent_dim=100, in_channel=1, im_size=32, hiddens=[512,256,128,64],init=2)])

class MabVAE(pl.LightningModule):
    def __init__(self,train_loader,decoders,eps=0.1,i=0):
        super(MabVAE, self).__init__()
        self.eps=eps
        self.decoders=decoders
        self.nb_decoders=len(decoders)
        self.latent_dim=self.decoders[0].latent_dim
        self.im_size = self.decoders[0].im_size
        self.in_channel=self.decoders[0].in_channel
        self.encoder = Encoder(latent_dim=self.latent_dim,in_channel=decoders[0].in_channel,im_size=self.im_size)
        self.history=torch.zeros(self.nb_decoders)
        self.NbDraws=torch.zeros(self.nb_decoders)
        self.strategy_path=[]
        self.train_loader = train_loader
        self.best_rewards=[]
        self.latent = torch.randn(64, self.latent_dim, 1, 1)
        self.i=i #counter of epochs
        self.t=1 #counter of steps

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick enabling to sample from N(mu, var) using
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar).type_as(mu)
        eps = torch.randn_like(std).type_as(mu)
        return eps * std + mu

    def loss_function(self, recons, input, mu, log_var):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """
        recon_loss = F.binary_cross_entropy(recons.view(-1, self.im_size, self.im_size), input.view(-1, self.im_size, self.im_size), reduction='sum')
        
        # minimize kl = maximize -kl
        weight = (len(self.train_loader.dataset) / (recons.size(0)))
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = weight * (recon_loss + kld_loss)

        return {'total_loss': loss, 'Reconstruction_Loss': weight * recon_loss, 'KL': weight * kld_loss}
        # Optimizers

    def configure_optimizers(self):
        opt_list=[]
        for decoder in self.decoders:
            optimizer = torch.optim.Adam(decoder.parameters())
            opt_list.append(optimizer)
            
        opt_list.append(torch.optim.Adam(self.encoder.parameters()))
        # return the list of optimizers and second empty list is for schedulers (if any)
        return opt_list, []

    # Calls after prepare_data for DataLoader
    def train_dataloader(self):
        return self.train_loader

    def forward(self, x):
        return self.encoder(x)

    # def display(self):
    #     fake = self.decoder(self.latent).detach()
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(np.transpose(utils.make_grid(fake, padding=2, normalize=True).cpu(), (1, 2, 0)))
    #     plt.savefig(f'VAE_current_result.png')
    #     print('OK')
    #     plt.close('all')

    # Training Loop
    def training_step(self, batch, batch_idx, optimizer_idx):
        # batch returns x and y tensors
        real_images, _ = batch
       
        #for the gpu
        if self.t==1:
            self.latent = self.latent.type_as(real_images[0])
            self.history=self.history.type_as(real_images[0])
            self.NbDraws=self.NbDraws.type_as(real_images[0])

        #encoding
        
        mu, log_var = self.encoder(real_images)
        mu=mu.type_as(real_images[0])
        log_var=log_var.type_as(real_images[0])
        z = self.reparameterize(mu, log_var).type_as(real_images[0])
        # print('OK1')
        with torch.no_grad():
            best_reward=10**(8)
            
            for decoder in self.decoders:
                
                recons=decoder(z).type_as(real_images[0])
                reward=self.loss_function(recons, real_images, mu, log_var)['Reconstruction_Loss']
                if reward<best_reward:
                    best_reward=reward

            self.best_rewards.append(best_reward)
        
        #initialization
        if self.t<self.nb_decoders:
            id_to_choose=self.t
            self.NbDraws[id_to_choose]+=1
        #eps-greedy behaviour
        else:
            u=np.random.random()
            if u<self.eps:
                id_to_choose=np.random.randint(self.nb_decoders)
                self.NbDraws[id_to_choose] += 1
            else:
                average_previous_rewards=self.history/self.NbDraws
                id_to_choose=torch.argmin(average_previous_rewards).item()
                self.NbDraws[id_to_choose] += 1

        # Encoder-Decoder
        recons = self.decoders[id_to_choose](z).type_as(real_images[0])
        step_dict = self.loss_function(recons, real_images, mu, log_var)
        
        self.history[id_to_choose]+=step_dict['Reconstruction_Loss']
        self.strategy_path.append(step_dict['Reconstruction_Loss'])
        total_loss = step_dict['total_loss']
        
        if self.t % 100 == 0:
            fake = self.decoders[id_to_choose](self.latent).detach()
            plt.figure(figsize=(10, 10))
            plt.imshow(np.transpose(utils.make_grid(fake, padding=2, normalize=True).cpu(), (1, 2, 0)))
            plt.savefig(f'VAE_current_result_decoder={id_to_choose}.png')
            plt.close('all')

        self.t += 1

        if (optimizer_idx in [id_to_choose,self.nb_decoders]):
            # {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
            output = OrderedDict({
                'loss': total_loss,
                'progress_bar': step_dict,
                'log': step_dict
            })

            return output

        
    # # calls after every epoch ends
    # def on_epoch_end(self):
    #     fake = self.decoder(self.latent).detach()
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(np.transpose(utils.make_grid(fake, padding=2, normalize=True).cpu(), (1, 2, 0)))
    #     plt.savefig(f'VAE_epoch_fashion_{self.i}.png')
    #     plt.close('all')
    #     self.i += 1