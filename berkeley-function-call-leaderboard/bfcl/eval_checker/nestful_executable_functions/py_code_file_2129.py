import numpy as np



def elementwise_mult(tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:

    """Performs element-wise multiplication between two tensors of equal dimension.

    The function supports broadcasting, and validates the input tensor dimensions and data types using assertions.

    Args:

        tensor1: The first tensor.

        tensor2: The second tensor.

    """

    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape"

    assert tensor1.dtype == tensor2.dtype, "Input tensors must have the same data type"

    return np.multiply(tensor1, tensor2)

