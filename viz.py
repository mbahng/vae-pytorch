import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
import torch
from model.vae import VAE
from model.cvae import CVAE

def reconstruct_vae(dataloader, model, savefig=None):
  """
  Compares original images to reconstructed for VAE models.

  Args:
    dataloader: DataLoader containing the images
    model: VAE model
    savefig: Path to save the figure. If None, displays the figure instead.
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

  if savefig is not None:
    plt.savefig(savefig)
    plt.close()
  else:
    plt.show()

def reconstruct_cvae(dataloader, model, savefig=None):
  """
  Compares original images to reconstructed for CVAE models.

  Args:
    dataloader: DataLoader containing the images and labels
    model: CVAE model
    savefig: Path to save the figure. If None, displays the figure instead.
  """
  x, y = next(iter(dataloader))
  x = x.squeeze(1)[:25]
  y = y[:25]
  xhat, _, _ = model(x.view(-1, 28 * 28), y)
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

  if savefig is not None:
    plt.savefig(savefig)
    plt.close()
  else:
    plt.show()

def reconstruct(dataloader, model, savefig=None):
  """
  Compares original images to reconstructed. Automatically detects model type.

  Args:
    dataloader: DataLoader containing the images (and labels for CVAE)
    model: VAE or CVAE model
    savefig: Path to save the figure. If None, displays the figure instead.

  Raises:
    NotImplementedError: If model is neither VAE nor CVAE
  """
  if isinstance(model, CVAE):
    reconstruct_cvae(dataloader, model, savefig=savefig)
  elif isinstance(model, VAE):
    reconstruct_vae(dataloader, model, savefig=savefig)
  else:
    raise NotImplementedError(f"Model type {type(model).__name__} is not supported. Expected VAE or CVAE.")

def interpolate(dataloader, model, savefig=None):
  """
  Visualize interpolations between images in latent space.
  You'll see that this interpolates much better than regular autoencoders.

  Args:
    dataloader: DataLoader containing the images and labels
    model: VAE or CVAE model
    savefig: Path to save the figure. If None, displays the figure instead.
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

  if savefig is not None:
    plt.savefig(savefig)
    plt.close()
  else:
    fig.show()

def sample_digits_vae(model, savefig=None):
  """
  Plots a 5x10 grid of 50 random samples generated from a VAE.

  Args:
    model: VAE model used to generate digits
    savefig: Path to save the figure. If None, displays the figure instead.
  """
  fig, axes = plt.subplots(5, 10, figsize=(15, 8))

  # Get the latent dimension from the model
  zdim = model.encoder.fc21.out_features

  # VAE: Generate random unconditioned samples
  for col in range(10):
    for row in range(5):
      # Sample random latent vector from standard normal
      z = torch.randn(1, zdim).to(model.device)

      # Generate image (no label conditioning)
      with torch.no_grad():
        x_generated = model.decode(z)

      # Reshape and convert to numpy for plotting
      img = x_generated.reshape(28, 28).cpu().numpy()

      # Plot the image
      axes[row, col].imshow(img, cmap='gray')
      axes[row, col].axis('off')

  plt.tight_layout()

  if savefig is not None:
    plt.savefig(savefig)
    plt.close()
  else:
    plt.show()

def sample_digits_cvae(model, savefig=None):
  """
  Plots a 5x10 grid of generated digits from a CVAE.
  Each column corresponds to a digit (0-9), with 5 samples per digit.

  Args:
    model: CVAE model used to generate digits
    savefig: Path to save the figure. If None, displays the figure instead.
  """
  fig, axes = plt.subplots(5, 10, figsize=(15, 8))

  # Get the latent dimension from the model
  zdim = model.encoder.fc21.out_features

  # CVAE: Generate samples conditioned on digit labels
  for col in range(10):  # 10 digits (0-9)
    for row in range(5):  # 5 samples per digit
      # Sample random latent vector from standard normal
      z = torch.randn(1, zdim).to(model.device)

      # Create label tensor for this digit
      y = torch.tensor([col]).to(model.device)

      # Generate image
      with torch.no_grad():
        x_generated = model.decode(z, y)

      # Reshape and convert to numpy for plotting
      img = x_generated.reshape(28, 28).cpu().numpy()

      # Plot the image
      axes[row, col].imshow(img, cmap='gray')
      axes[row, col].axis('off')

      # Add column title for the first row
      if row == 0:
        axes[row, col].set_title(f'{col}', fontsize=12)

  plt.tight_layout()

  if savefig is not None:
    plt.savefig(savefig)
    plt.close()
  else:
    plt.show()

def sample_digits(model, savefig=None):
  """
  Plots a 5x10 grid of generated digits. Automatically detects model type.

  For VAE: Generates 50 random samples from the latent space.
  For CVAE: Each column corresponds to a digit (0-9), with 5 samples per digit.

  Args:
    model: VAE or CVAE model used to generate digits
    savefig: Path to save the figure. If None, displays the figure instead.

  Raises:
    NotImplementedError: If model is neither VAE nor CVAE
  """
  if isinstance(model, CVAE):
    sample_digits_cvae(model, savefig=savefig)
  elif isinstance(model, VAE):
    sample_digits_vae(model, savefig=savefig)
  else:
    raise NotImplementedError(f"Model type {type(model).__name__} is not supported. Expected VAE or CVAE.")

def visualize_latent_space_vae(dataloader, model, savefig=None):
  """
  Visualizes the 2D latent space for a VAE model.
  Each digit is plotted with a different color.

  Args:
    dataloader: DataLoader containing the images and labels
    model: VAE model with 2D latent space
    savefig: Path to save the figure. If None, displays the figure instead.
  """
  model.eval()

  # Collect all latent representations and labels
  z_list = []
  labels_list = []

  with torch.no_grad():
    for x, y in dataloader:
      x = x.view(x.size(0), -1).to(model.device)
      mu, _ = model.encode(x)
      z_list.append(mu.cpu().numpy())
      labels_list.append(y.numpy())

  # Concatenate all batches
  z = np.concatenate(z_list, axis=0)
  labels = np.concatenate(labels_list, axis=0)

  # Plot
  plt.figure(figsize=(10, 8))
  scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
  plt.colorbar(scatter, ticks=range(10), label='Digit')
  plt.xlabel('Latent Dimension 1')
  plt.ylabel('Latent Dimension 2')
  plt.title('VAE Latent Space Visualization')
  plt.grid(True, alpha=0.3)

  if savefig is not None:
    plt.savefig(savefig)
    plt.close()
  else:
    plt.show()

def visualize_latent_space_cvae(dataloader, model, savefig=None):
  """
  Visualizes the 2D latent space for a CVAE model.
  Each digit is plotted with a different color.

  Args:
    dataloader: DataLoader containing the images and labels
    model: CVAE model with 2D latent space
    savefig: Path to save the figure. If None, displays the figure instead.
  """
  model.eval()

  # Collect all latent representations and labels
  z_list = []
  labels_list = []

  with torch.no_grad():
    for x, y in dataloader:
      x = x.view(x.size(0), -1).to(model.device)
      y = y.to(model.device)
      mu, _ = model.encode(x, y)
      z_list.append(mu.cpu().numpy())
      labels_list.append(y.cpu().numpy())

  # Concatenate all batches
  z = np.concatenate(z_list, axis=0)
  labels = np.concatenate(labels_list, axis=0)

  # Plot
  plt.figure(figsize=(10, 8))
  scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
  plt.colorbar(scatter, ticks=range(10), label='Digit')
  plt.xlabel('Latent Dimension 1')
  plt.ylabel('Latent Dimension 2')
  plt.title('CVAE Latent Space Visualization')
  plt.grid(True, alpha=0.3)

  if savefig is not None:
    plt.savefig(savefig)
    plt.close()
  else:
    plt.show()

def visualize_latent_space(dataloader, model, savefig=None):
  """
  Visualizes the 2D latent space. Automatically detects model type.
  Each digit is plotted with a different color.

  Args:
    dataloader: DataLoader containing the images and labels
    model: VAE or CVAE model with 2D latent space
    savefig: Path to save the figure. If None, displays the figure instead.

  Raises:
    ValueError: If latent space is not 2D
    NotImplementedError: If model is neither VAE nor CVAE
  """
  # Check if latent space is 2D
  zdim = model.encoder.fc21.out_features
  if zdim != 2:
    raise ValueError(f"Cannot visualize latent space with dimension {zdim}. Only 2D latent spaces can be visualized.")

  if isinstance(model, CVAE):
    visualize_latent_space_cvae(dataloader, model, savefig=savefig)
  elif isinstance(model, VAE):
    visualize_latent_space_vae(dataloader, model, savefig=savefig)
  else:
    raise NotImplementedError(f"Model type {type(model).__name__} is not supported. Expected VAE or CVAE.")



