import os 
import torch
import argparse
import numpy as np
from utils import *
from models import *
from torch import optim
from time import strftime
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import datasets, transforms


def train( model, optimizer, epoch, 
           train_loader, loss_function, 
           device, filename = 'model', log_interval = 1000):
    '''
    Training generative model
    '''
    model.train()
    
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
           epoch, train_loss / len(train_loader.dataset)))

def test( epoch, model, device, directory):
    '''
    Visually inspect results of trained model
    by sampling from latent space gaussian distribution.
    '''
    model.eval()
    with torch.no_grad():
        
        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   directory + '/sample_' + str(epoch) + '.png')
        
def main():

    # Experimental Settings
    parser = argparse.ArgumentParser(description='PyTorch CNP Experiment')
    
    parser.add_argument('--batch-size', type = int, default = 16, metavar = 'N',
                        help = 'Input batch size for training (default: 16)')
    
    parser.add_argument('--model', default = 'VAE', metavar = 'M',
                        help = 'Model [VAE, CBVAE] (default: VAE)')
    
    parser.add_argument('--path', default = '../data', metavar = 'T',
                        help = 'Path to dataset')
    
    parser.add_argument('--epochs', type = int, default = 100, metavar = 'N',
                        help = 'Number of epochs to train (default: 100)')
    
    parser.add_argument('--lr', type = float, default = 1e-3, metavar = 'LR',
                        help = 'Learning rate (default: 1e-3)')
    
    parser.add_argument('--warp', type = float, default = 0., metavar = 'W',
                        help = 'Warping constant (default: 0.)')
    
    parser.add_argument('--seed', type = int, default = 512, metavar = 'S',
                         help ='Random seed (default: 1)')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help ='Disables CUDA training')
    
    parser.add_argument('--report-interval', type = int, default = 1, metavar = 'REP',
                        help ='Epochs to wait before storing trained model')
         
    parser.add_argument('--log-interval', type = int, default = 500, metavar = 'LOG',
                        help ='Batches to wait before logging training status')
    
    # Cheeck all training settings
    args = parser.parse_args()
    print('Experimental Settings \n')
    for arg in vars(args):
        print(arg, getattr(args, arg))
        
    # Create folders to store experiment results
    directory = 'results/' + args.model
    if not os.path.exists(directory):
        os.makedirs(directory)
   
    data_directory = 'trained_models/' + args.model
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    
    timestamp = strftime("%Y-%m-%d_%H-%M")
    run_stamp = data_directory + '/run_' + str(timestamp)
        
    # Write configuration 
    file = open(run_stamp + '.txt','w') 
    file.write('Experimental Settings\n')
    for arg in vars(args):
        file.write(str(arg) + ' ' + str(getattr(args, arg)) + '\n')
    file.close() 
    
    # Check if GPU available for training
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Training using ', device)
    
    # Set reproducibility seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Load MNIST train dataset
    mnist_train_loader = torch.utils.data.DataLoader(
                            datasets.MNIST('../data', train=True, download=True,
                            transform = transforms.Compose([AddNoiseToTensor(), 
                                                            Warping(args.warp)])),
                            batch_size = args.batch_size, shuffle = True)

    
    # Choose the model version
    model = VAE().to(device)
    
    if args.model == 'CBVAE':
        loss = loss_cbvae
    elif args.model == 'VAE':
        loss = loss_vae
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    
    # Training and Test Loop    
    for epoch in range(1, args.epochs + 1):
             
            # Train
            train(model, optimizer, epoch, mnist_train_loader, loss, device, 
                  filename = args.model, log_interval = args.log_interval)
            
            # Test
            if epoch % args.report_interval == 0:
                
                # Store model
                torch.save(model.state_dict(), run_stamp + '_epoch_' + str(epoch) + '.pt' )
                
                # Visualize Results
                test(epoch, model, device, directory)


if __name__ == '__main__':
    main()
