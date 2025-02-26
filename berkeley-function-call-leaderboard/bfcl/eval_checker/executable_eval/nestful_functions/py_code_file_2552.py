from typing import List



def remove_less_or_equal_to_3(numbers: List[int]) -> List[int]:

    """Removes every element from a list of numbers whose value is less than or equal to 3.



    Args:

        numbers: A list of numbers.



    Returns:

        A new list containing only the elements from the input list that are greater than 3.

    """

    return [i for i in numbers if i > 3]

