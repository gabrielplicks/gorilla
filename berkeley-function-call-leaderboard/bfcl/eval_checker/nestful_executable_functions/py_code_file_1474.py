from typing import Union



def convert_nanosecond_to_microsecond(nanosecond: Union[int, float]) -> int:

    """Converts a nanosecond duration to microseconds rounded down to the nearest integer.

    Args:

        nanosecond: The duration in nanoseconds.

    """

    microseconds = nanosecond // 1000

    return microseconds

