from typing import List, Dict, Any



def filter_non_empty(input_list: List[Any]) -> List[Any]:

    """

    Filters a list of objects and returns a list of objects with only non-empty values.

    Args:

        input_list: A list of objects.

    Raises:

        ValueError: If the input list contains invalid elements.

    """

    output_list = []

    for element in input_list:

        if isinstance(element, dict):

            if any(value != '' for value in element.values()):

                output_list.append(element)

        elif isinstance(element, str):

            if element != '':

                output_list.append(element)

        else:

            raise ValueError('Invalid element in the input list.')

    return output_list

