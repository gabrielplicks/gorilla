def interval_selection(n: int) -> str:

    """

    Returns an interval selection based on the value of `n`.

    Args:

        n: The input number.

    """

    if n < 0:

        return '< 0'

    elif 0 <= n < 10:

        return '0 <= n < 10'

    elif 10 <= n < 20:

        return '10 <= n < 20'

    else:

        return '>= 20'

