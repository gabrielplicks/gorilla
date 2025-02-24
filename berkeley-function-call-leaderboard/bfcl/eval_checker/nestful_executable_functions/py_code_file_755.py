from typing import List



def convert_string_to_array(string: str) -> List[int]:

    """Converts a string of digits, separated by commas and/or spaces, into an array of integers.

    Args:

        string: The input string.

    """

    string = "".join(string.split(","))

    string = "".join(string.split(" "))

    return [int(s) for s in string]

