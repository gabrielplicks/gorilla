from typing import List



def format_with_type(input_list: List[str], type: str) -> List[str]:

    """Formats a list of strings according to different types.



    Args:

        input_list: The list of strings to format.

        type: The type of format to apply.



    Returns:

        A new list with the formatted strings.



    Raises:

        ValueError: If the type is not supported.

    """

    if type == "uppercase":

        return [word.upper() for word in input_list]



    if type == "lowercase":

        return [word.lower() for word in input_list]



    if type == "sentence":

        return [word.capitalize() for word in input_list]



    if type == "title":

        return [word.title() for word in input_list]



    raise ValueError("Invalid type: ", type)

