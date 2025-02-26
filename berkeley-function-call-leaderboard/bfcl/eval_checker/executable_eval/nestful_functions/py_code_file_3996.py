from typing import Union



def _format(s: str) -> str:

    """Formats a string by replacing 'e' with 'E', splitting on 'E', and replacing '.' with 'e' in the second part.

    Args:

        s: The input string.

    Returns:

        The formatted string.

    """

    s = s.replace("e", "E")

    parts = s.split("E")

    if len(parts) == 2:

        first_part, second_part = parts

        if "." not in second_part:

            return s

        first_part, second_part = second_part.split(".")

        if "." not in first_part:

            return s

        first_part = first_part.replace(".", "e")

        if "." not in second_part:

            return s

        s = f"{first_part}E{second_part}"

    return s

