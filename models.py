import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    
    # VAE model 

    ## Architectured Based on Appendix by Authors
    ## https://arxiv.org/src/1907.06845v4/anc/cont_bern_aux.pdf

    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder layers
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc21 = nn.Linear(500, 20)
        self.fc22 = nn.Linear(500, 20)
        
        # Decoder layers
        self.fc3 = nn.Linear(20, 500)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, 784)
        
        # Dropout Layers
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)

    def encode(self, x):
        #Recognition function
        h1 = F.relu(self.fc1(x))
        h1 = self.dropout1(h1)
        h2 = F.relu(self.fc2(h1))
        h2 = self.dropout2(h2)
        return self.fc21(h2), F.softplus(self.fc22(h2)) 

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        #Likelihood function
        h3 = F.relu(self.fc3(z))
        h3 = self.dropout3(h3)
        h4 = F.relu(self.fc4(h3))
        h4 = self.dropout4(h4)
        return torch.sigmoid( self.fc5(h4) ) # Gaussian mean

    def forward(self, x):
        mu, std = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, std)
        # Return decoding, mean and logvar
        return self.decode(z), mu, 2.*torch.log(std) 