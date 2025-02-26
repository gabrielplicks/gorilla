from typing import List, Tuple



def even_odd_sums(numbers: List[int]) -> Tuple[int, int]:

    """Computes the sum of the even numbers and the sum of the odd numbers in a given list.

    Args:

        numbers: A list of integers.

    Returns:

        A tuple containing the sum of even numbers and the sum of odd numbers.

    """

    even_sum = 0

    odd_sum = 0

    for number in numbers:

        if number % 2 == 0:

            even_sum += number

        else:

            odd_sum += number

    return (even_sum, odd_sum)

