from typing import Union



def calculate_label(value: Union[int, float], threshold: Union[int, float]) -> int:

    """Calculates a label given a value and a threshold.

    Args:

        value: The value to compare with the threshold.

        threshold: The threshold value.

    """

    if value < threshold:  # Check if value is less than threshold

        label = 0

    else:  # value is greater than or equal to threshold

        label = 1

    return label

