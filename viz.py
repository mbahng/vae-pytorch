import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
import torch

def reconstruct(dataloader, model): 
  """
  Compares original images to reconstructed. 
  Nothing too special since regular AEs can do the same. 
  """
  x, _ = next(iter(dataloader)) 
  x = x.squeeze(1)[:25] 
  xhat, _, _ = model(x.view(-1, 28 * 28))
  xhat = xhat.reshape(-1, 28, 28).detach().cpu().numpy()
  x = x.detach().cpu().reshape(25, 28, 28).numpy() 


  fig = plt.figure(figsize=(15, 8))
  subfigs = fig.subfigures(nrows=1, ncols=2)

  axes_left = subfigs[0].subplots(nrows=5, ncols=5)
  for i in range(25): 
    axes_left[i // 5, i % 5].imshow(x[i], cmap="gray")
    axes_left[i // 5, i % 5].axis('off')
  subfigs[0].suptitle('Original')

  axes_right = subfigs[1].subplots(nrows=5, ncols=5)
  for i in range(25): 
    axes_right[i // 5, i % 5].imshow(xhat[i], cmap="gray")
    axes_right[i // 5, i % 5].axis('off')
  subfigs[1].suptitle('Reconstructed')
  plt.show()

def interpolate(dataloader, model): 
  """
  Visualize interpolations between images in latent space. 
  You'll see that this interpolates much better than regular autoencoders. 
  """

  X_test, Y_test = next(iter(dataloader))
  X_batch, Y_batch = X_test[:128], Y_test[:128]
  data_size = X_test.size()
  data = X_test.view(X_test.size(0),-1).cpu()
  Z, _ = model.encode(data)

  def get_centroid(x):
    """
    Computes the centroid of images in the latent space.
    Args: x: torcch.Tensor of shape: batch x 1 x 28 x 28
    Returns: z_centroid: Centroid in latent space.
    """
    data = x.view(x.size(0),-1).cpu()
    Z, _ = model.encode(data)
    Z_centroid = Z.mean(axis=0)
    return Z_centroid

  def get_a2b(a_label: int, b_label: int):
      """Computes the vector in latent space from centroid of a to centroid of b.

      Args:
          a_label: Class `a`
          b_label: Class `b`

      Returns:
          z_a2b: Vector from centroid of `a` to centroid of `b`.
      """

      x_a = X_test[Y_test == a_label]
      x_b = X_test[Y_test == b_label]

      z_a = get_centroid(x_a)
      z_b = get_centroid(x_b)
      z_a2b = z_b - z_a
      return z_a2b

  def interpolate_(a_label = 0):
      """Interpolate in latent space from one class to another class."""

      all_classes = np.arange(0, 10)
      all_classes = np.delete(all_classes, a_label)
      z_a2b_all = []
      for b_label in all_classes:
          z_a2b_all.append(get_a2b(a_label, b_label))

      x_a = X_test[Y_test == a_label]
      data = x_a.view(x_a.size(0),-1).cpu()
      z_a, _ = model.encode(data)
      z_in = z_a[0]

      x_interpolated = []
      for z_a2b in z_a2b_all:
          for alpha in np.arange(0, 2, 0.2):
              z = z_in + alpha*z_a2b
              x_vae = model.decode(z).detach()
              x_interpolated.append(x_vae)

      nrow = len(x_interpolated)
      x_all = torch.stack(x_interpolated)
      img = make_grid(x_all.reshape((nrow, 1, 28, 28)), padding=0, nrow=nrow//9)
      npimg = img.cpu().numpy()

      return npimg
  fig, axes = plt.subplots(2, 5, figsize=(20, 10))

  for a_label, ax in enumerate(axes.flat):
      img = interpolate_(a_label=a_label)
      ax.imshow(np.transpose(img, (1,2,0)), interpolation='nearest')

  fig.show()

