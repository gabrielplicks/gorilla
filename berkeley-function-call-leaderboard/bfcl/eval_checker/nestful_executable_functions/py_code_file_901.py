import numpy



def shape_to_strings(tensor: numpy.ndarray) -> list[str]:

    """Converts the shape of a 3-D tensor to a list of strings.



    Args:

        tensor: The input 3-D tensor.



    Returns:

        A list of strings that represent the shape of the tensor.

    """

    shape = tensor.shape

    shape_strings = []

    for i, dimension in enumerate(shape, 1):

        shape_strings.append(f"{dimension} is the {i} dimension")



    return shape_strings

