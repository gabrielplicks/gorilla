from typing import Union



def get_n_term_of_fibonacci_sequence(n: int) -> Union[int, float]:

    """Calculates the nth term of the Fibonacci sequence.

    Args:

        n: The position of the term in the Fibonacci sequence.

    """

    if n == 1:

        return 0

    elif n == 2:

        return 1

    else:

        return get_n_term_of_fibonacci_sequence(n - 1) + get_n_term_of_fibonacci_sequence(n - 2)

