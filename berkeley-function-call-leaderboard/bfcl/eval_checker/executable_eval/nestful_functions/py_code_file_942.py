from typing import Any, List



def convert_list_to_list(input_list: List[Any]) -> List[Any]:

    """

    Converts a list of lists to a list of lists of integers. Converts each

    element to an integer if it's a string, or each element of the internal list

    to an integer if it's a list of strings.

    Args:

        input_list: The input list of lists.

    """

    output_list = []



    for element in input_list:

        if isinstance(element, str):

            output_list.append(int(element))

        elif isinstance(element, list):

            output_list.append(convert_list_to_list(element))

        else:

            output_list.append(element)



    return output_list

