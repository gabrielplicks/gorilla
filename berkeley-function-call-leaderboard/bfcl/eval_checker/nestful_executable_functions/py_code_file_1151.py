from typing import List, Union



def get_largest_element(input_list: List[int], use_builtin: bool) -> Union[int, float]:

    """

    Returns the largest element in a list.

    If `use_builtin` is `True`, uses the built-in `max` function;

    otherwise, implements custom logic to find the largest element.



    Args:

        input_list: A list of integers.

        use_builtin: A boolean value indicating whether to use the built-in `max` function.

    """

    if use_builtin:

        return max(input_list)

    else:

        largest_element = float('-inf')

        for element in input_list:

            if element > largest_element:

                largest_element = element

        return largest_element

