from typing import List, Union



def trend_and_retrend(sequence: List[float], trend: Union[int, float], start: float) -> List[float]:

    """Performs trending and retrending on a sequence of data points.



    Args:

        sequence: A sequence of real numbers.

        trend: The trend value.

        start: The start value.



    Raises:

        Exception: If the trend value is zero.

    """

    if trend == 0:

        raise Exception("Trend cannot be zero.")



    new_sequence = []

    for i, data in enumerate(sequence):

        new_sequence.append(start + trend * i)



    return new_sequence, trend

