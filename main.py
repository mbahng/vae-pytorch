import torch
from model import VAE
from traintest import * 
from dataset import *
import viz 

device = "cpu"

train_ds, test_ds = mnist_flat()

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

model = VAE(idim=28 * 28, hdim=400, zdim=20, device=device)
# model = CVAE(feature_dim=28 * 28, class_dim=10, hdim=400, zdim=2, device=device)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

for epoch in range(1, 31):
  train(model, train_dl, optimizer)
  test(model, test_dl)

torch.save(model.state_dict(), f"saved/model-ep30.pt")


# Visualization
viz.sample_digits(model, savefig="fig/vae_sample_digits.png")
viz.reconstruct(test_dl, model, savefig="fig/vae_reconstruct.png")
viz.interpolate(test_dl, model, savefig="fig/vae_interpolate.png")

viz.visualize_latent_space(test_dl, model, savefig="fig/vae_latent_space.png")
