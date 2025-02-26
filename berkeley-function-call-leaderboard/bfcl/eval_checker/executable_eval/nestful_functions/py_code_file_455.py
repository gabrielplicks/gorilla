import torch



def flatten_spatial_dims(x: torch.Tensor) -> torch.Tensor:

    """Reshapes the input tensor by flattening all spatial dimensions into a single dimension.



    Args:

        x: A tensor with a shape of `(N, C, H, W)`.



    Returns:

        A tensor with a shape of `(N, C * H * W)`.

    """

    batch_size, num_channels, height, width = x.shape

    return x.reshape(batch_size, num_channels * height * width)

