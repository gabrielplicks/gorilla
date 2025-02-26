from typing import List



def filter_complex_numbers(complex_numbers: List[complex], n: int) -> List[complex]:

    """

    Filters a list of complex numbers and returns the first `n` complex numbers

    whose real and imaginary parts are both integers, sorted in ascending order.



    Args:

        complex_numbers: A list of complex numbers.

        n: The number of complex numbers to return.



    Returns:

        A list of the first `n` complex numbers whose real and imaginary parts

        are both integers, sorted in ascending order.

    """

    filtered_numbers = [

        num for num in complex_numbers

        if num.real.is_integer() and num.imag.is_integer()

    ]

    sorted_numbers = sorted(filtered_numbers, key=lambda num: abs(num))

    return sorted_numbers[:n]

