from torchvision import transforms 
import torchvision
import torch
from model import VAE
from traintest import * 
import viz 

batch_size = 128
device = "cpu"

train_set,test_set,train_loader,test_loader = {},{},{},{}
transform = transforms.Compose(
    [transforms.ToTensor()])

train_ds = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
test_ds = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

model = VAE(28 * 28, 400, 20).to(device) 

model.load_state_dict(torch.load("saved/model-ep30.pt"))

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

torch.save(model.state_dict(), f"saved/model-ep0.pt")

for epoch in range(1, 31):
  train(model, device, train_dl, optimizer, epoch)
  test(model, device, test_dl, epoch)
  if (epoch) % 10 == 0:
    torch.save(model.state_dict(), f"saved/model-ep{epoch}.pt")

# Visualization
viz.reconstruct(test_dl, model)
viz.interpolate(test_dl, model)
