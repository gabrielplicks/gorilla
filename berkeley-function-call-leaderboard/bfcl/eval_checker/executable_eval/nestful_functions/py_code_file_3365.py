import numpy as np



def perspective_projection(point: np.array, K: np.array) -> np.array:

    """

    Performs perspective projection of a 3D point (x, y, z) using a camera matrix (K) and returns its 2D projection (x', y') in the image plane.



    Args:

        point: A numpy array representing a 3D point (x, y, z).

        K: A numpy array representing the camera matrix.



    Returns:

        A numpy array representing the projected 2D point (x', y') in the image plane.

    """

    assert point.shape == (3,) and K.shape == (3, 3), "Incorrect input shape"

    result = K @ point

    x_proj = result[0] / result[2]

    y_proj = result[1] / result[2]



    return np.array([x_proj, y_proj])

