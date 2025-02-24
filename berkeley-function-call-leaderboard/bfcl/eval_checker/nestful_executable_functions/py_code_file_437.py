from typing import Union



def bool_indicator(x: Union[int, float], threshold: Union[int, float]) -> int:

    """Calculates the boolean indicator of a real number `x` with respect to a threshold.



    Args:

        x: A real number.

        threshold: A real number between 0 and 1.



    Returns:

        Either 0 or 1, according to the formula:



        $$

        bool\_indicator(x, threshold) = \begin{cases}

            1 & \text{if } x \ge threshold \\

            0 & \text{otherwise}

        \end{cases}

        $$

    """

    return round(x >= threshold)

