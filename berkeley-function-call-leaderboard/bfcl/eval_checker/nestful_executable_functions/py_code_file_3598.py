import numpy as np

import typing



def create_tensor_from_key(dictionary: typing.Union[dict, list], key: str) -> np.ndarray:

    """Creates a tensor from the values of a specific key in a dictionary with nested key-value pairs.

    The tensor is a stacked collection of all the tensors created from the values of the key.

    Args:

        dictionary: The dictionary with nested key-value pairs.

        key: The key whose values will be used to create the tensor.

    """

    stack = []

    def traverse(d):

        if isinstance(d, dict):

            for k, v in d.items():

                if k == key:

                    stack.append(np.array(v))

                else:

                    traverse(v)

        elif isinstance(d, list):

            for element in d:

                traverse(element)

    traverse(dictionary)

    return np.stack(stack)

