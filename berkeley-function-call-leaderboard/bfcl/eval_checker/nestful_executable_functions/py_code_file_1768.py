from typing import List



def less_than_max_value(input_list: List[int], max_value: int) -> List[str]:

    """

    Returns a list of integers from the input list that are less than the max_value.

    Each integer in the returned list is preceded by a comment indicating whether it is odd or even.



    Args:

        input_list: A list of integers.

        max_value: The maximum value for the integers in the returned list.



    Returns:

        A list of strings, where each string is a comment indicating whether the integer is odd or even,

        followed by the integer itself.

    """

    result = ["# " + ("odd" if i % 2 == 1 else "even") + ": " + str(i) for i in input_list if i < max_value]

    return result

