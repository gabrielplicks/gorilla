from typing import Dict, Union



def colorize_text(text: str, color: Union[str, None]) -> str:

    """

    Applies the appropriate ANSI escape code for the specified color to the input text.

    Args:

        text: The input text to be colorized.

        color: The color to apply to the text. Must be one of "red", "green", or "blue".

    """

    color_codes: Dict[str, str] = {

        "red": "\033[31m",

        "green": "\033[32m",

        "blue": "\033[34m",

    }

    if color not in color_codes:

        raise ValueError(f"Invalid color: {color}")

    return f"{color_codes[color]}{text}\033[0m"

