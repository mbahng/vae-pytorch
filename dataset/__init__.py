from torchvision import transforms 
import torchvision
import torch

def mnist_flat(): 
  transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Lambda(lambda x: torch.flatten(x))
  ])

  train_ds = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
  test_ds = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
  return train_ds, test_ds
