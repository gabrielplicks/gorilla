import numpy as np



def construct_tensor() -> np.ndarray:

    """

    Constructs a 3-dimensional tensor of shape (2, 4, 3) filled with random floating-point values between 0 and 1.

    Then, assigns the value 42 to the element at the first row, second column, and third plane.



    Returns:

        The constructed tensor.

    """

    try:

        tensor = np.random.random((2, 4, 3))

        tensor[0, 1, 2] = 42



        return tensor

    except IndexError:

        print("Index out of bounds")

