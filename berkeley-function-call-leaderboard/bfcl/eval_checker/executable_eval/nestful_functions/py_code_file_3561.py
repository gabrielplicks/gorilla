from typing import List



def concatenate_to_string(input_list: List[str], prefix: str = "", suffix: str = "") -> str:

    """Concatenates the values of a list to a string with a prefix and suffix.

    Args:

        input_list: The list of values to be concatenated.

        prefix: The prefix string to be concatenated to the beginning of the output string.

        suffix: The suffix string to be concatenated to the end of the output string.

    """

    result = prefix + ", ".join(str(element) for element in input_list) + suffix

    return result

