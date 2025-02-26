import torch



def torch_sigmoid_function(input: torch.Tensor) -> torch.Tensor:

    """Applies the sigmoid function to each element of the input tensor.



    Args:

        input: The input tensor.



    Returns:

        A tensor of the same shape as the input tensor, with the sigmoid function applied to each element.

    """

    return torch.sigmoid(input)

