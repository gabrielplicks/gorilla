from typing import List



def sum_of_square_of_even_numbers(nums: List[int]) -> int:

    """Calculates the sum of the square of each even number in a list of integers.

    Args:

        nums: A list of integers.

    """

    return sum(pow(num, 2) for num in filter(lambda x: x % 2 == 0, nums))

