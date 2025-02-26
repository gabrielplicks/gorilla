from typing import List



def formatted_output(strings: List[str]) -> str:

    """

    Formats a list of strings in a formatted way.



    Args:

        strings: A list of strings to be formatted.



    Returns:

        A formatted string with each string in the input list prefixed with its index and a period.

    """

    output = []

    for index, string in enumerate(strings, start=1):

        output.append(f"{index}. {string}")

    return "\n".join(output)

