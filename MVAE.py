import torch
from torch import nn
from torch.nn import functional as F


# Build Encoder
class Encoder(pl.LightningModule):

    def __init__(self, latent_dim,im_size=64,in_channel=3,hiddens=[64,128,256,512]):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.im_size=im_size
        self.in_channel=in_channel
        self.hiddens=hiddens
        self.modules=[nn.Conv2d(in_channel,hiddens[0], kernel_size=3, stride=2, padding=1),  # hiddens[0]*im_size/2xim_size/2
                      nn.ReLU(),
                      nn.BatchNorm2d(hiddens[0])]

        for i in range(1,len(hiddens)):
            # hiddens[i]*(im_size/2^i)^2
            self.modules.append(nn.Conv2d(hiddens[i-1],hiddens[i], kernel_size=3, stride=2, padding=1)
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

        for i in range(1,len(hiddens)-1):
            self.module.append(nn.ReLU())
            self.modules.append(nn.Linear(hiddens[i-1],hiddens[i]))

        self.modules.append(nn.Tanh())
        self.modules.append(nn.Linear(hiddens[-1],self.im_size*self.im_size))
        self.modules.append(nn.Sigmoid())

        self.decode=nn.Sequential(*self.modules)

    def forward(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        out = self.decode(z)
        out = out.view(-1,self.in_channel,self.im_size,self.im_size)

        return out

# Build Decoder
class Decoder_Conv(pl.LightningModule):

    def __init__(self, latent_dim=4, in_channel=3, im_size=64, hiddens=[512,256,128,64],init=4):
        super(Decoder_MLP, self).__init__()

        assert (2**(len(hiddens))*init==im_size),"Take care about the architecture of your decoder, there is something wrong"

        self.latent_dim = latent_dim
        self.im_size = im_size
        self.hiddens=hiddens
        self.in_channel = in_channel
        self.modules = [nn.ConvTranspose2d(self.latent_dim,hiddens[0],init, 1, 0), #init*init
                       nn.ReLU(),
                       nn.BatchNorm2d(self.im_size * 8)]

        for i in range(1, len(hiddens) - 1):
            self.modules.append(nn.ConvTranspose2d(hiddens[i-1],hiddens[i-1], 4, 2, 1)) #im_size=2^(2+i)
            self.modules.append(nn.ReLU())
            self.modules.append(nn.BatchNorm2d(hiddens[i]))


        self.modules.append(nn.ConvTranspose2d(hiddens[-1],self.in_channel, 4, 2, 1)) #im_size*im_size
        self.modules.append(nn.Sigmoid())

        self.decode = nn.Sequential(*self.modules)

    def forward(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        z = z.view(-1,self.latent_dim, 1, 1)
        result = self.decode(z)

        return result


# Build Decoder
class Decoder_Linear_Conv(pl.LightningModule):

    def __init__(self, latent_dim=4, in_channel=3, im_size=64, hiddens=[512,256,128,64],init=4):
        super(Decoder_MLP, self).__init__()

        assert (2**(len(hiddens))*init==im_size),"Take care about the architecture of your decoder, there is something wrong"

        self.latent_dim = latent_dim
        self.im_size = im_size
        self.hiddens= hiddens
        self.in_channel = in_channel
        self.modules = [nn.Linear(self.latent_dim,self.latent_dim//2),
                        nn.ReLU(),
                        nn.ConvTranspose2d(self.latent_dim,hiddens[0],init, 1, 0), #init*init
                        nn.SELU()]

        for i in range(1, len(hiddens) - 1):
            self.modules.append(nn.ConvTranspose2d(hiddens[i-1],hiddens[i-1], 4, 2, 1)) #im_size=2^(2+i)
            self.modules.append(nn.SELU())

        self.modules.append(nn.ConvTranspose2d(hiddens[-1],self.in_channel, 4, 2, 1)) #im_size*im_size
        self.modules.append(nn.Sigmoid())

        self.decode = nn.Sequential(*self.modules)

    def forward(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        z = z.view(-1,self.latent_dim, 1, 1)
        result = self.decode(z)

        return result


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0, 0.08)
        nn.init.constant_(m.bias.data, 0)

Decoders=[Decoder_MLP(latent_dim=100, in_channel=3, im_size=64, hiddens=[256,128,64,32]),
          Decoder_Conv(latent_dim=100, in_channel=3, im_size=64, hiddens=[512,256,128],init=8),
          Decoder_Linear_Conv(latent_dim=100, in_channel=3, im_size=64, hiddens=[512,256,128,64],init=4)]

class MVAE(pl.LightningModule):

    def __init__(self,train_loader,decoders,i=0):
        super(MVAE, self).__init__()
        self.encoder = Encoder(self.latent_dim)
        self.decoders=decoders
        self.latent_dim=self.decoders[0].latent_dim
        self.im_size = self.decoders[0].im_size
        self.in_channel=self.decoders[0].in_channel
        self.train_loader = train_loader
        # self.fixed_noise = torch.randn(64, self.latent_dim, 1, 1)
        self.i = i

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick enabling to sample from N(mu, var) using
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
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
        optimizerE = torch.optim.Adam(self.encoder.parameters(), lr=2 * 10 ** (-4), betas=(0.5, 0.9))
        optimizerD = torch.optim.Adam(self.decoder.parameters(), lr=2 * 10 ** (-4), betas=(0.5, 0.9))
        # return the list of optimizers and second empty list is for schedulers (if any)
        return [optimizerE, optimizerD], []

    # Calls after prepare_data for DataLoader
    def train_dataloader(self):
        return self.train_loader

    def forward(self, x):
        return self.encoder(x)

    def display(self):
        fake = self.decoder(self.fixed_noise).detach()
        plt.figure(figsize=(10, 10))
        plt.imshow(np.transpose(utils.make_grid(fake, padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.savefig(f'VAE_current_result.png')
        print('OK')
        plt.close('all')

    # Training Loop
    def training_step(self, batch, batch_idx, optimizer_idx):
        # batch returns x and y tensors
        real_images, _ = batch
        self.cpt += 1

        self.fixed_noise = self.fixed_noise.type_as(real_images[0])
        # if self.cpt==0:

        #   self.logscale=self.logscale.type_as(real_images[0])

        mu, log_var = self.encoder(real_images)
        z = self.reparameterize(mu, log_var)

        # Encoder-Decoder
        recons = self.decoder(z)

        step_dict = self.loss_function(recons, real_images, mu, log_var)

        total_loss = step_dict['total_loss']

        # Encoder
        # {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
        output = OrderedDict({
            'loss': total_loss,
            'progress_bar': step_dict,
            'log': step_dict
        })

        if self.cpt % 100 == 0:
            fake = self.decoder(self.fixed_noise).detach()
            plt.figure(figsize=(10, 10))
            plt.imshow(np.transpose(utils.make_grid(fake, padding=2, normalize=True).cpu(), (1, 2, 0)))
            plt.savefig(f'VAE_current_result.png')
            plt.close('all')

        return output

    # calls after every epoch ends
    def on_epoch_end(self):
        fake = self.decoder(self.fixed_noise).detach()
        plt.figure(figsize=(10, 10))
        plt.imshow(np.transpose(utils.make_grid(fake, padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.savefig(f'VAE_epoch_fashion_{self.i}.png')
        plt.close('all')
        self.i += 1