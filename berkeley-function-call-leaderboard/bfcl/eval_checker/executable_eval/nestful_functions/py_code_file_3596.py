import numpy as np



def get_height_and_width(input_tensor: np.ndarray) -> tuple:

    """Inspects a given image tensor and returns the height and width separately.



    Args:

        input_tensor: The input tensor to inspect.



    Raises:

        TypeError: If input_tensor is not a numpy array.

        ValueError: If the input tensor does not have the expected shape.

    """

    if not isinstance(input_tensor, np.ndarray):

        raise TypeError("Input tensor must be a numpy array")

    if len(input_tensor.shape) != 4:

        raise ValueError("Invalid shape: expected a 4D tensor of shape (batch_size, height, width, num_channels)")

    height, width = input_tensor.shape[1], input_tensor.shape[2]

    return height, width

