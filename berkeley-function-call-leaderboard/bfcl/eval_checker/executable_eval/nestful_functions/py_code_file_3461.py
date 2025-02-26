import numpy as np



def expand_and_replace(x: np.ndarray) -> np.ndarray:

    """Expands the dimension of a batch of inputs and replaces negative values with 0.



    Given a batch of inputs `x` with shape `(batch_size, input_dim)`, the function

    returns a batch of inputs with shape `(batch_size, 1)`.



    The function also replaces any negative values in the inputs with 0.



    Args:

        x: The input batch of shape `(batch_size, input_dim)`.



    Returns:

        A batch of inputs with shape `(batch_size, 1)` and negative values replaced with 0.

    """

    x_expanded = np.expand_dims(x, axis=1)

    x_expanded[x_expanded < 0] = 0

    return x_expanded

