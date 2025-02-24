import numpy as np



def unfold_tensor(X: np.ndarray) -> np.ndarray:

    """Unfolds a 3D tensor into a 2D matrix.



    Args:

        X: A 3D tensor of shape (I, J, K).



    Returns:

        A 2D matrix of shape (I * J, K).

    """

    X_unfolded = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))

    return X_unfolded

