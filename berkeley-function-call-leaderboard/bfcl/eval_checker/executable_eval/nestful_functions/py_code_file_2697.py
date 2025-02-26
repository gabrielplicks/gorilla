from typing import List



def remove_repetition(numbers: List[int]) -> List[int]:

    """Removes repetition from a list of numbers.



    Args:

        numbers: A list of numbers with repetition.



    Returns:

        A new list containing the same numbers, but with each number appearing only once.

    """

    unique_numbers = set()



    for num in numbers:

        unique_numbers.add(num)



    return list(unique_numbers)

