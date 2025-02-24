import torch



def calculate_derivatives(x1: torch.Tensor, x2: torch.Tensor) -> tuple:

    """Calculates the derivatives of a custom function f with respect to x1 and x2 using PyTorch.



    Args:

        x1: The first tensor.

        x2: The second tensor.



    Returns:

        A tuple containing the derivative of f with respect to x1 and x2.

    """

    condition = x1 < x2

    derivative_x1 = torch.where(condition, torch.ones_like(x1), torch.ones_like(x1))

    derivative_x2 = torch.where(condition, -torch.ones_like(x2), torch.ones_like(x2))

    return derivative_x1, derivative_x2

