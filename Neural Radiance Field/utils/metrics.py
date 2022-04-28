import torch

def PSNR(image_pred: torch.Tensor, image_gt: torch.Tensor) -> torch.Tensor:
    """Peak signal noise ratio."""
    return -10.0 * torch.log10(torch.mean(torch.square(image_pred, image_gt)))
