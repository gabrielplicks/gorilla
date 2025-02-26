import numpy as np



def linear_regression_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """Calculates the loss function for a linear regression model using a cumulative sum approach.



    Args:

        y_true: The true target values.

        y_pred: The predicted values.



    Returns:

        The cumulative sum of the residuals squared.

    """

    residuals = y_true - y_pred

    squared_residuals = [residual ** 2 for residual in residuals]

    return np.sum(squared_residuals)

