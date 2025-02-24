from typing import Dict, Any



def type_checking(input: Any) -> str:

    """

    Checks if the input is a dict, and if it is, determines whether it is a dict of str to int.

    Args:

        input: The input to check.

    Returns:

        A string indicating whether the input is a dict of str to int or not.

    """

    input_type = type(input)

    if not isinstance(input, Dict):

        raise TypeError(f"Expected a dict, got {input_type}")

    is_dict_str_to_int = True

    for key, value in input.items():

        if not isinstance(key, str) or not isinstance(value, int):

            is_dict_str_to_int = False

            break



    if is_dict_str_to_int:

        return "The input is a dict of str to int"

    else:

        return "The input is a dict, but not a dict of str to int"

