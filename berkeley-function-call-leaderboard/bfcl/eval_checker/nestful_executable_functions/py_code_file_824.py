import torch



def relu_inplace_torch(tensor: torch.Tensor) -> torch.Tensor:

    """Performs inplace ReLU operation on a 2D tensor given as input using torch.



    Args:

        tensor: The input 2D tensor.



    Returns:

        The updated tensor after applying the inplace ReLU operation.

    """

    if not isinstance(tensor, torch.Tensor) or not tensor.dim() == 2:

        print("Invalid input. Please provide a valid 2D tensor.")

        return None

    tensor.clamp_(min=0)

    return tensor

