from typing import List, Union



def replace_tabs(string: str) -> str:

    """Replaces any tab characters in a string with spaces.



    Args:

        string: The input string.



    Returns:

        The modified string with tab characters replaced by spaces.

    """

    result: List[Union[str, int]] = []

    for char in string:

        if char == '\t':

            result.append(' ' * 4)  # Replace tab characters with 4 spaces

        else:

            result.append(char)



    return ''.join(result)

