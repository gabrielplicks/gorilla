from typing import Dict



def split_string_into_parts_and_convert_to_dictionary(string: str, delimiter: str) -> Dict[int, str]:

    """Splits a string into multiple parts based on a given delimiter and converts it into a dictionary with the indices as keys and the parts as values.



    Args:

        string: The input string to be split.

        delimiter: The delimiter used to split the string.



    Returns:

        A dictionary with the indices as keys and the parts as values.

    """

    parts = string.split(delimiter)

    parts_dict = {index: part for index, part in enumerate(parts)}

    return parts_dict

