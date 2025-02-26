import numpy as np



def make_3d(x: np.ndarray, n: int) -> np.ndarray:

    """Creates a 3D tensor with the same values as `x` and the shape `(x.shape[0], x.shape[1], n)`.



    Args:

        x: The input 2D tensor.

        n: The integer to determine the third dimension of the output tensor.



    Returns:

        The 3D tensor with the same values as `x` and the shape `(x.shape[0], x.shape[1], n)`.

    """

    y = np.empty((x.shape[0], x.shape[1], n))

    for i in range(n):

        y[:, :, i] = x



    return y

