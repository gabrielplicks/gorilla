from typing import AnyStr



def frame_string(text: AnyStr, border: AnyStr) -> AnyStr:

    """Converts a string into a framed string with a unique border character.



    Args:

        text: The input string to be framed.

        border: The unique border character to be used.

    """

    width = len(text) + 2

    border_chars = [border] * width

    frame = border + text + border + "\n"

    return frame

