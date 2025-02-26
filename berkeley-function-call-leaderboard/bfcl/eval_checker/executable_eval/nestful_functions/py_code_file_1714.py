import numpy as np



def orthogonal_projection(point: np.ndarray, axis: np.ndarray) -> np.ndarray:

    """Computes the orthogonal projection of a 3D point onto a given axis.



    Args:

        point: The 3D point to project.

        axis: The axis onto which the point is projected.



    Raises:

        ValueError: If point and axis are not numpy arrays or lists.

    """

    if isinstance(point, list) and isinstance(axis, list):

        point = np.array(point)

        axis = np.array(axis)

    elif not isinstance(point, np.ndarray) or not isinstance(axis, np.ndarray):

        raise ValueError('Point and axis must be numpy arrays or lists.')



    A = point

    B = axis

    return A - (np.dot(A, B) / np.dot(B, B)) * B

