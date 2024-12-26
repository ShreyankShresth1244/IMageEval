import torch
import torch.nn.functional as F


def compute_clarity(image_tensor):
    """
    Compute clarity using a Laplacian filter on the image tensor.

    Args:
        image_tensor (torch.Tensor): Input image tensor of shape (C, H, W).

    Returns:
        torch.Tensor: Clarity result as a scalar.
    """
    # Define the Laplacian kernel
    kernel = torch.tensor([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]], dtype=torch.float32, device=image_tensor.device).unsqueeze(0).unsqueeze(0)

    # Add batch dimension if necessary
    if image_tensor.dim() == 3:  # Shape (C, H, W)
        image_tensor = image_tensor.unsqueeze(0)  # Convert to (N, C, H, W)

    # Apply the Laplacian kernel
    clarity = F.conv2d(image_tensor, kernel, padding=1)

    # Compute clarity score as mean of absolute values
    return clarity.abs().mean().item()
