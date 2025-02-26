import torch

import numpy as np



def numpy_to_torch(numpy_array: np.ndarray) -> torch.Tensor:

    """Converts a Numpy array to a PyTorch tensor with the same shape and values.

    The function converts the data type to a corresponding PyTorch data type and

    performs a clamp operation on values to fall within the range [-1, 1].



    Args:

        numpy_array: The input Numpy array.



    Returns:

        A PyTorch tensor with the same shape and values as the input Numpy array.

    """

    shape = numpy_array.shape

    data_type = torch.float32 if numpy_array.dtype == "float32" else torch.int64

    torch_tensor = torch.tensor(numpy_array.tolist(), dtype=data_type)

    torch_tensor.clamp_(-1, 1)



    return torch_tensor

