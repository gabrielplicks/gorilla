from typing import Union



def modified_string(text: Union[str, None]) -> str:

    """Modifies a string based on whether it starts with "Hello".



    Args:

        text: The input string.



    Returns:

        A modified string.

    """

    if text is None:

        return "Hello, world!"

    if text.startswith("Hello"):

        return "Hello, world!"

    else:

        return f"Hello, {text}!"

