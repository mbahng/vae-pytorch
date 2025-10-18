import torch.nn.functional as F
import torch
from model import VAE
from torch.utils.data import DataLoader
from torch.optim import Optimizer

def loss_function(recon_x, x, mu, logvar):
  """Computes the loss = -ELBO = Negative Log-Likelihood + KL Divergence.
      Args:
      recon_x: Decoder output.
      x: Ground truth.
      mu: Mean of Z
      logvar: Log-Variance of Z
      p(z) here is the standard normal distribution with mean 0 and identity covariance.
  """
  BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') # BCE = -Negative Log-likelihood
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # KL Divergence b/w q_\phi(z|x) || p(z)
  return BCE + KLD

def train(model: VAE, device: str, train_loader: DataLoader, optimizer: Optimizer, epoch: int):
  train_loss = 0
  model.train()
  for data, _ in train_loader:
    data = data.view(data.size(0),-1).to(device)

    output, mu, logvar = model(data)

    loss = loss_function(output, data, mu, logvar)
    train_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f'Train Epoch ({epoch}): Loss: {train_loss / len(train_loader.dataset)}') # type: ignore

def test(model: VAE, device: str, test_loader: DataLoader, epoch: int):
  model.eval()
  test_loss = 0
  with torch.no_grad():
    for data, _ in test_loader:
      data = data.view(data.size(0),-1).to(device)

      output, mu, logvar = model(data)

      loss = loss_function(output, data, mu, logvar)
      test_loss += loss.item() 

  print(f'Test Epoch ({epoch}): Loss: {test_loss / len(test_loader.dataset)}') # type: ignore

