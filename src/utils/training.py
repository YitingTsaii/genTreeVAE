# ==========================================
# TRAINING & VALIDATION LOOPS
# ==========================================

import torch
from torch.nn import functional as F_nn
from ..config import LOG_INTERVAL


def loss_function_Fmat(recon_x, x, mu, logvar, struct_loss, constraint_weight):
    """
    Computes the Fmat-aware loss for the Phylogenetic VAE.

    Args:
        recon_x (Tensor): Reconstructed flattened F-matrix (batch_size, input_dim).
        x (Tensor): Original input flattened F-matrix (batch_size, input_dim).
        mu (Tensor): Mean of the latent Gaussian distribution.
        logvar (Tensor): Log-variance of the latent Gaussian distribution.
        struct_loss (Tensor): Scalar loss from FMatrixLayer representing constraint violations.
        constraint_weight (float): Hyperparameter scaling the importance of structural validity.

    Returns:
        tuple: (total_loss, recon_loss, kl_loss, weighted_struct_loss)
               Returns individual components for logging/monitoring.
    """
    recon_loss = F_nn.mse_loss(recon_x, x, reduction='sum')  # MSE loss for reconstruction
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
    weighted_struct_loss = constraint_weight * struct_loss
    total_loss = recon_loss + kl_loss + weighted_struct_loss
    return total_loss, recon_loss, kl_loss, weighted_struct_loss


def train(model, train_loader, optimizer, device, epoch, latent_dim, constraint_weight):
    """
    Training loop for one epoch.

    Args:
        model: The VAE model
        train_loader: DataLoader for training data
        optimizer: Optimizer instance
        device: Device to use (cpu or cuda)
        epoch: Current epoch number
        latent_dim: Latent dimension (for KL per-dim calculation)
        constraint_weight: Weight for structural loss

    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    train_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0
    train_struct_loss = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, struct_loss = model(data)
        loss, recon_loss, kl_loss, weighted_struct_loss = loss_function_Fmat(
            recon_batch, data, mu, logvar, struct_loss, constraint_weight=constraint_weight
        )
        loss.backward()
        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()
        train_struct_loss += weighted_struct_loss.item()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_recon_loss = train_recon_loss / len(train_loader.dataset)
    avg_train_kl_loss = train_kl_loss / len(train_loader.dataset)
    avg_train_struct_loss = train_struct_loss / len(train_loader.dataset)

    # KL per latent dim - for diagnosing collapse across latent dim
    avg_kl_per_dim = avg_train_kl_loss / latent_dim

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, avg_train_loss))
    return avg_train_loss


def validate(model, val_loader, device, epoch, latent_dim, constraint_weight):
    """
    Validation loop.

    Args:
        model: The VAE model
        val_loader: DataLoader for validation data
        device: Device to use (cpu or cuda)
        epoch: Current epoch number
        latent_dim: Latent dimension (for KL per-dim calculation)
        constraint_weight: Weight for structural loss

    Returns:
        float: Average validation loss
    """
    model.eval()
    val_loss = 0
    val_recon_loss = 0
    val_kl_loss = 0
    val_struct_loss = 0

    with torch.no_grad():
        for (data, targets) in val_loader:
            data = data.to(device)
            recon, mu, logvar, struct_loss = model(data)

            loss, recon_loss, kl_loss, weighted_struct_loss = loss_function_Fmat(
                recon, data, mu, logvar, struct_loss, constraint_weight=constraint_weight
            )
            val_loss += loss.item()
            val_recon_loss += recon_loss.item()
            val_kl_loss += kl_loss.item()
            val_struct_loss += weighted_struct_loss.item()

    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_recon_loss = val_recon_loss / len(val_loader.dataset)
    avg_val_kl_loss = val_kl_loss / len(val_loader.dataset)
    avg_val_struct_loss = val_struct_loss / len(val_loader.dataset)
    avg_kl_per_dim = avg_val_kl_loss / latent_dim

    print(f'====> Validation set loss: {avg_val_loss:.4f} (Recon: {avg_val_recon_loss:.4f}, KL: {avg_val_kl_loss:.4f})')
    return avg_val_loss


def test(model, test_loader, device, latent_dim, constraint_weight):
    """
    Testing loop.

    Args:
        model: The VAE model
        test_loader: DataLoader for test data
        device: Device to use (cpu or cuda)
        latent_dim: Latent dimension (for KL per-dim calculation)
        constraint_weight: Weight for structural loss

    Returns:
        float: Average test loss
    """
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    test_struct_loss = 0

    with torch.no_grad():
        for (data, targets) in test_loader:
            data = data.to(device)
            recon, mu, logvar, struct_loss = model(data)

            loss, recon_loss, kl_loss, weighted_struct_loss = loss_function_Fmat(
                recon, data, mu, logvar, struct_loss, constraint_weight=constraint_weight
            )
            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_kl_loss += kl_loss.item()
            test_struct_loss += weighted_struct_loss.item()

    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_test_recon_loss = test_recon_loss / len(test_loader.dataset)
    avg_test_kl_loss = test_kl_loss / len(test_loader.dataset)
    avg_test_struct_loss = test_struct_loss / len(test_loader.dataset)
    avg_kl_per_dim = avg_test_kl_loss / latent_dim

    print(f'====> Test set loss: {avg_test_loss:.4f} (Recon: {avg_test_recon_loss:.4f}, KL: {avg_test_kl_loss:.4f})')
    return avg_test_loss
