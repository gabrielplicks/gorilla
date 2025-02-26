from typing import Union



def pad_with_binary_char(text: str, char: Union[str, int]) -> str:

    """Pads the input text with a specific binary character at both the beginning and end.



    Args:

        text: The input string that needs to be padded.

        char: The binary character used for padding.

    """

    return str(char) + text + str(char)

