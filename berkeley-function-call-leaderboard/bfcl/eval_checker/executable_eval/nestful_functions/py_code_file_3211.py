from typing import List, Union



def first_odd(input_list: List[int]) -> Union[int, None]:

    """Returns the first odd number in the input list.



    Args:

        input_list: A list of integers.



    Returns:

        The first odd number in the input list, or None if there is no odd number.

    """

    odd_list = [num for num in input_list if num % 2 == 1]

    return odd_list[0] if len(odd_list) != 0 else None

