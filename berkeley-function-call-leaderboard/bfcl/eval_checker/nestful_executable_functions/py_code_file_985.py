from typing import List, Union



def toggle_values(input_list: List[Union[int, bool]]) -> List[Union[int, bool]]:

    """Toggles the values in a list of integers or booleans.



    Inverts all the boolean values, while leaving the integers unchanged.



    Args:

        input_list: A list of integers or booleans.



    Raises:

        ValueError: If the input is not a list or if any element is not an integer or a boolean.

    """

    if not isinstance(input_list, list):

        raise ValueError("Input must be a list")

    for element in input_list:

        if not isinstance(element, (int, bool)):

            raise ValueError("Each element must be an integer or a boolean")

    for i in range(len(input_list)):

        if isinstance(input_list[i], bool):

            input_list[i] = not input_list[i]

    return input_list

