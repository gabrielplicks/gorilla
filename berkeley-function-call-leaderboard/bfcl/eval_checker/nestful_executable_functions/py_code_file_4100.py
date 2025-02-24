from typing import List



def get_sum_of_squares_of_evens(nums: List[int]) -> int:

    """Calculates the sum of the squares of the even numbers in a list of integers.



    Args:

        nums: A list of integers.



    Raises:

        TypeError: If the input list contains non-integer elements.

    """

    total_sum = 0



    for num in nums:

        if not isinstance(num, int):

            raise TypeError("Input list must contain only integers.")



        if num % 2 == 0:

            total_sum += num * num



    return total_sum

