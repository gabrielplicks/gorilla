import torch



def add_noise_to_torch_tensor(tensor: torch.Tensor, scale: float, device: torch.device) -> torch.Tensor:

    """Adds Gaussian noise to a PyTorch tensor.



    Args:

        tensor: The tensor to which noise is to be added.

        scale: The standard deviation of the noise distribution.

        device: The device on which the noise tensor will be created.

    """

    noise = torch.randn(tensor.shape, device=device) * scale

    return tensor + noise

