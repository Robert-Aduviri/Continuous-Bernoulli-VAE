import torch
from torch.nn import functional as F

def sumlogC( x , eps = 1e-5):
    '''
    Numerically stable implementation of 
    sum of logarithm of Continous Bernoulli
    constant C, using Taylor 2nd degree approximation
        
    Parameter
    ----------
    x : Tensor of dimensions (batch_size, dim)
        x takes values in (0,1)
    ''' 
    x = torch.clamp(x, eps, 1.-eps) 
    mask = torch.abs(x - 0.5).ge(eps)
    far = torch.masked_select(x, mask)
    close = torch.masked_select(x, ~mask)
    far_values =  torch.log( (torch.log(1. - far) - torch.log(far)).div(1. - 2. * far) )
    close_values = torch.log(torch.tensor((2.))) + torch.log(1. + torch.pow( 1. - 2. * close, 2)/3. )
    return far_values.sum() + close_values.sum()

def loss_vae(recon_x, x, mu, logvar):
    '''
    Variational Autoencoder Loss Function
    Described by https://github.com/pytorch/examples/blob/master/vae/main.py
    '''
    BCE = F.binary_cross_entropy(recon_x[0], x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def loss_cbvae(recon_x, x, mu, logvar):
    '''
    Loss function for continuous bernoulli vae
    '''
    BCE = F.binary_cross_entropy(recon_x[0], x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    LOGC = -sumlogC(recon_x[0])
    return BCE + KLD + LOGC

def loss_gvae(recon_x, x, mu, logvar):
    '''
    Loss function for Gaussian vae
    Described by https://github.com/atinghosh/VAE-pytorch/blob/master/VAE_CNN_Gaussianloss.py
    '''
    x = x.view(-1, 784)
    mu_x, sigma = recon_x    
    GLL = torch.sum(torch.log(sigma)) + 0.5 * torch.sum(((x - mu_x) / sigma).pow(2)) 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD + GLL

class AddNoiseToTensor(object):
    '''
    MNIST Preprocessing as described in Appendix 4

    Add custom transformation that adds uniform [0,1] noise 
    to the integer pixel values between 0 and 255 and then 
    divide by 256, to obtain values in [0,1]
    '''
    def __call__(self, pic):
        
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        tensor = img.float()
        return ( tensor + torch.rand(tensor.size())).div(256.)
    
    
def warp(gamma, x):
    '''
    Warping transform
    
    Parameter
    ----------
    gamma : float 
        warping constant in (-0.5, 0.5)
    
    x : Tensor of dimensions (batch_size, dim)
        x takes values in (0,1)
    ''' 
    if gamma == -0.5:
        return x.ge(0.5).float()
    elif gamma > -0.5 and gamma < 0:
        return torch.clamp( ( x + gamma ).div(1. + 2. * gamma ), 0., 1.)
    elif gamma >= 0 and gamma <= 0.5:
        return gamma + ( 1. - 2. * gamma) * x 

    
class Warping(object):
    '''
    Warping transformation
    '''
    def __init__(self, parameter):
        self.parameter = parameter
            
    def __call__(self, pic):
        return warp(self.parameter, pic)
