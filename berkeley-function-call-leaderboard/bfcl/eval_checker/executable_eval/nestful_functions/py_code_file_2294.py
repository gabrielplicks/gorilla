from typing import Union



def calculate_error(y_true: Union[int, float], y_pred: Union[int, float]) -> Union[int, float]:

    """Calculates the error between a predicted value and the actual value.

    Args:

        y_true: The actual value.

        y_pred: The predicted value.

    """

    error = abs(y_true - y_pred)

    return error

