from typing import Dict, List



def prefix_list(input_list: List[str], prefix: str, mapping: Dict[str, str] = None) -> List[str]:

    """Converts a list of strings into a list of strings with a prefix.

    If a string in the input list is in the mapping dictionary, it should be replaced with the corresponding value from the dictionary.

    Args:

        input_list: The input list of strings.

        prefix: The prefix to add to each string.

        mapping: A dictionary mapping strings to their replacements.

    """

    mapping = mapping or {}

    output_list = []

    for item in input_list:

        if item in mapping:

            item = mapping[item]

        output_list.append(prefix + str(item))

    return output_list

