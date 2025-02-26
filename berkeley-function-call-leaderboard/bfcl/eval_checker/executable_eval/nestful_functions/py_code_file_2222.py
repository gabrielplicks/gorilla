from typing import Union



def calc_precision(tp: Union[int, float], fp: Union[int, float]) -> Union[float, int]:

    """Calculates the precision metric given true positive and false positive values.



    Args:

        tp: The true positive value.

        fp: The false positive value.



    Returns:

        The precision score, which is the ratio of `tp` to the sum of `tp` and `fp`. If the sum is 0, the function returns `-1`.

    """

    if tp == 0 and fp == 0:

        return -1

    return tp / (tp + fp)

