import math



def filter_non_numbers(input_list: list) -> list:

    """Filters a list to only include numbers.



    Args:

        input_list: The list to filter.



    Returns:

        A new list containing only numbers.

    """

    output_list = []

    for element in input_list:

        if isinstance(element, (int, float)) and not math.isnan(element) and not math.isinf(element):

            output_list.append(element)

    return output_list

