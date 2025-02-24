import torch



def matrix_product_without_allocation(X: torch.Tensor) -> torch.Tensor:

    """Computes the product X @ X.T without allocating any new tensors.



    Args:

        X: The input sparse tensor.



    Returns:

        The result of X @ X.T as a tensor.

    """

    return X @ X.T

