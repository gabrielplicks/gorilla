import numpy as np



def elementwise_product_with_broadcasting(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:

    """Calculates the element-wise product of two tensors, allowing for broadcasting to take place.

    The returned tensor is always of the same shape as the first input tensor.

    Args:

        t0: The first tensor.

        t1: The second tensor.

    """

    if t0.shape == t1.shape:

        return t0 * t1

    else:

        t1_broadcast = np.broadcast_to(t1, t0.shape)

        return t0 * t1_broadcast

