import numpy as np



def pyflow_loss(flow1: np.ndarray, flow2: np.ndarray) -> float:

    """Computes the loss between two optical flow predictions.



    The loss function is defined as the mean squared error between the two flows.



    Args:

        flow1: The actual flow, a numpy array of shape (height, width, 2).

        flow2: The predicted flow, a numpy array of shape (height, width, 2).



    Returns:

        A single float value representing the loss.

    """

    flow_diff = flow1 - flow2

    flow_diff_squared = np.square(flow_diff)

    return np.mean(flow_diff_squared)

