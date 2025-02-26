from typing import Union



def soft_thresholding(x: Union[int, float], lambda_: Union[int, float]) -> Union[int, float]:

    """Performs the soft thresholding operation on a given value `x` with a threshold `lambda`.



    Args:

        x: The value to apply the soft thresholding operation to.

        lambda_: The threshold value.

    """

    if x > lambda_:

        return x - lambda_

    elif x < -lambda_:

        return x + lambda_

    else:

        return 0

