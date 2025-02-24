from typing import List, Tuple



def even_numbers_with_indices(numbers: List[int]) -> Tuple[List[int], List[int]]:

    """

    Returns a tuple containing a list of even numbers from the input list and a list of their indices.



    Args:

        numbers: A list of integers.



    Returns:

        A tuple containing a list of even numbers and a list of their indices.

    """

    even_numbers = []

    indices = []



    for i, num in enumerate(numbers):

        if num % 2 == 0:

            even_numbers.append(num)

            indices.append(i)

    return even_numbers, indices

