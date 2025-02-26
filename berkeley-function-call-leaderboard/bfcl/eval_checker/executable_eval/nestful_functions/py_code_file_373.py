from typing import List, Tuple

import torch



def shape_info(tensor: torch.Tensor) -> List[int]:

    """Retrieves the shape information of a PyTorch tensor.



    Args:

        tensor: The input PyTorch tensor.



    Returns:

        A list containing the size of each dimension of the tensor.

    """

    shape_list = []

    for dim in tensor.shape:

        shape_list.append(int(dim))

    return shape_list

