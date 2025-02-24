import numpy as np



def concat_new_axis(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    """Concatenates tensors `x` and `y` over a new axis with the goal of reshaping them from being one-dimensional to two-dimensional.



    Args:

        x: The first tensor to be concatenated.

        y: The second tensor to be concatenated.



    Returns:

        The concatenated tensors `x` and `y` as a two-dimensional tensor.

    """

    x_with_new_axis = np.expand_dims(x, axis=1)

    y_with_new_axis = np.expand_dims(y, axis=1)

    return np.concatenate([x_with_new_axis, y_with_new_axis], axis=1)

