import random

random.seed(42)
import typing



def rand_int_between(low: int, high: typing.Optional[int] = None) -> int:

    """Returns a random integer between `low` and `high`. If `high` is not specified, the range is from 0 to `low`.



    Args:

        low: The lower bound of the range.

        high: The upper bound of the range. If not provided, the range is from 0 to `low`.

    """

    if high is None:

        high = low

        low = 0



    rand_int = random.randint(low, high)

    return rand_int

