from typing import Union



def round_to_value(x: Union[int, float], y: Union[int, float], strategy: str) -> Union[int, float]:

    """Rounds a value to a specified value using a specified rounding strategy.



    Args:

        x: The value to be rounded.

        y: The value to round to.

        strategy: The rounding strategy to use. Can be either "ceil", "floor", or "nearest".



    Raises:

        ValueError: If the rounding strategy is invalid.

    """

    if strategy == "ceil":

        return y * (x // y + 1) if x % y != 0 else x

    elif strategy == "floor":

        return y * (x // y)

    elif strategy == "nearest":

        return y * round(x / y)

    else:

        raise ValueError("Invalid rounding strategy")

