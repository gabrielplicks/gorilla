from typing import List, Any



def flatten_list_dynamic(input_list: List[List[Any]]) -> List[Any]:

    """Flattens a list of lists using dynamic programming.



    Args:

        input_list: The list of lists to be flattened.



    Returns:

        A flattened list.

    """

    output_list = []

    for sublist in input_list:

        for element in sublist:

            output_list.append(element)

    return output_list

