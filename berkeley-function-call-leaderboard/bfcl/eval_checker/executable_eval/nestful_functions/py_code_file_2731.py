import torch



def swish_activation(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:

    """

    Applies the Swish activation function element-wise to the input tensor.



    Args:

        x: The input tensor.

        beta: The slope of the sigmoid function.



    Returns:

        The output tensor after applying the Swish activation function.

    """

    return x * torch.sigmoid(beta * x)

