import torch

def PSNR(image_pred: torch.Tensor, image_gt: torch.Tensor) -> torch.Tensor:
    """Peak signal noise ratio."""
    return -10.0 * torch.log10((image_pred - image_gt).square().mean())

def MSE2PSNR(mse: torch.Tensor) -> torch.Tensor:
    """Convert mse loss to PSNR."""
    return -10.0 * torch.log10(mse)
