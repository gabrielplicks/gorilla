from typing import List



def concatenate_with_modifications(strings: List[str]) -> str:

    """Concatenates a list of strings with the following modifications:

    - The first character of each string is capitalized

    - A comma is added after each string except the last

    - The last string ends with a period.



    Args:

        strings: A list of strings to be concatenated.



    Returns:

        A string that is the concatenation of the input strings with the specified modifications.

    """

    output = ""

    for index, string in enumerate(strings):

        if index == 0:

            output += string[0].upper() + string[1:]

        else:

            output += ", " + string

    output += "."

    return output

