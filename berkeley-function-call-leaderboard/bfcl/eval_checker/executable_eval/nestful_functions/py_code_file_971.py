import torch

import numpy as np



def any_rows_equal(tensor: torch.Tensor) -> bool:

    """Determines if any two rows are equal in a PyTorch tensor.



    Args:

        tensor: A PyTorch tensor of shape `(n, 2)`.



    Returns:

        True if any two rows are equal, False otherwise.

    """

    array = tensor.numpy()

    n = array.shape[0]



    for i in range(n):

        for j in range(i + 1, n):

            if (array[i] == array[j]).all():

                return True



    return False

