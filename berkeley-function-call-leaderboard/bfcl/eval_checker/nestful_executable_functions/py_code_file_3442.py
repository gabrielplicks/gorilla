from typing import List



def get_string_numbers(nums: List[int]) -> str:

    """Constructs a string of numbers from a list of numbers.



    For each element in the list, adds "n element x" to the string, where `x` is replaced by the element and `n` is the index of the element plus one.

    If the list is empty, the function returns an empty string.



    Args:

        nums: A list of numbers.

    """

    size = len(nums)

    result = ""

    for i in range(size):

        string = f"{i + 1} element {nums[i]}"

        result += string + "\n"



    return result

