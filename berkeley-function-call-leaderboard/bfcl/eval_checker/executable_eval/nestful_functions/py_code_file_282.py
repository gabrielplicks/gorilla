import torch



def move_tensor(tensor: torch.Tensor, use_cuda: bool) -> torch.Tensor:

    """Moves a PyTorch tensor to the CUDA device if the `use_cuda` flag is `True` and CUDA is available.



    Args:

        tensor: The PyTorch tensor to move.

        use_cuda: A boolean flag indicating whether to move the tensor to the CUDA device.



    Returns:

        The tensor moved to the selected device.

    """

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    tensor = tensor.to(device)

    print(f"Tensor is on device: {device.type}, moved: {tensor.device != device}")

    return tensor

