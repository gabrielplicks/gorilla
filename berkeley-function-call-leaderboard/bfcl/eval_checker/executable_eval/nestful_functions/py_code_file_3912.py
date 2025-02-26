from typing import List



def replace_separators(input_list: List[str]) -> List[str]:

    """Replaces `_` with `-` and vice versa in a list of strings.



    If a string is already separated by `_` or `-`, then the function removes the separators and returns the string as is.



    Args:

        input_list: A list of strings.



    Returns:

        A list of strings with `_` replaced with `-` and vice versa.

    """

    output_list = []

    for string in input_list:

        if "_" in string:

            output_list.append(string.replace("_", "-"))

        elif "-" in string:

            output_list.append(string.replace("-", "_"))

        else:

            output_list.append(string)

    return output_list

