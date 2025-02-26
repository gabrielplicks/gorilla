from typing import List, Union



def convert_to_printable_str(input_list: List[Union[int, float, str]]) -> str:

    """Converts a list of numbers or strings to a printable string.



    Args:

        input_list: A list of numbers or strings.



    Returns:

        A string representation of the input list.

    """

    if not input_list:

        return ''

    output_list = []

    for item in input_list:

        if isinstance(item, int) or isinstance(item, float):

            output_list.append(str(item))

        else:

            output_list.append(item)

    return ', '.join(output_list)

