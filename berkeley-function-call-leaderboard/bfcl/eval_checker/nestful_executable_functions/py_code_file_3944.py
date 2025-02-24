from typing import List

import numpy as np



def get_shape_in_format(t: np.ndarray) -> List[int]:

    """Returns the shape of the tensor in the format [batch, height, width, channels].



    Args:

        t: The input tensor.



    Returns:

        The shape of the tensor in the format [batch, height, width, channels].

    """

    t_shape = t.shape

    batch = t_shape[0]

    height = t_shape[1]

    width = t_shape[2]

    channels = t_shape[3]



    return [batch, height, width, channels]

