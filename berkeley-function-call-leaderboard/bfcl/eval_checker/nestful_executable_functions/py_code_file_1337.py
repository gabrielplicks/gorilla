import torch

from typing import List



def get_first_three_not_zero(x: torch.Tensor) -> List[float]:

    """

    Returns a list of the first three elements of `x` that are not equal to zero.

    Creates entries in the output list at any indices where `x` is equal to zero.

    If `x` has fewer than three non-zero elements, fill the rest of the list with zeros.



    Args:

        x: The input tensor.



    Raises:

        ValueError: If `x` is not a tensor.

    """

    if not isinstance(x, torch.Tensor):

        raise ValueError("Input must be a tensor.")



    output = []



    for element in x:

        if element != 0:

            output.append(element)

        else:

            output.append(0)



        if len(output) == 3:

            return output



    output += [0] * (3 - len(output))

    return output

