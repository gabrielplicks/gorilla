from typing import List, Any



def format_sequence(elements: List[Any], format_string: str) -> List[str]:

    """Formats a sequence of elements using a provided format string.



    Args:

        elements: The sequence of elements to format.

        format_string: The format string to use for formatting.

    """

    formatted_elements = []

    for element in elements:

        formatted_elements.append(format_string.format(element))

    return formatted_elements

