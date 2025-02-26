from typing import Union



MAX_UNSIGNED_INT: int = 2**32 - 1  # Maximum value of a 32-bit unsigned integer



def diff_ticks(prev_tick: Union[int, float], curr_tick: Union[int, float]) -> int:

    """

    Computes the difference between two time ticks, assuming the timer runs continuously without overflow.

    The difference is represented as a 32-bit unsigned integer.

    Args:

        prev_tick: The previous time tick.

        curr_tick: The current time tick.

    """

    if curr_tick < prev_tick:

        curr_tick += MAX_UNSIGNED_INT + 1

    diff = curr_tick - prev_tick



    return diff

