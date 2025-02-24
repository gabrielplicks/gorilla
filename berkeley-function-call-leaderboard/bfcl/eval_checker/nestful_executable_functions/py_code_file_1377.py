import numpy as np



def compute_l2_distance(X: np.ndarray) -> np.ndarray:

    """Computes the L2 distance between each pair of points in `X` and stores the result in a NumPy array `D`.



    Args:

        X: A vectorized NumPy array of size (n, 3) containing 3D points.



    Returns:

        A NumPy array `D` of size (n, n) containing the L2 distances between each pair of points in `X`.

    """

    D = np.zeros((X.shape[0], X.shape[0]))



    for i in range(X.shape[0]):

        for j in range(X.shape[0]):

            D[i, j] = np.linalg.norm(X[i] - X[j])



    return D

