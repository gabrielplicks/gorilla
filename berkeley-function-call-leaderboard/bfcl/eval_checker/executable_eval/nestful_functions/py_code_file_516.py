from typing import List, Union



def trim_indentation(text: str) -> str:

    """Removes leading spaces from each line in the given text.



    The function removes any leading spaces from each line, and trims the trailing spaces on the line.

    It handles the case where the input contains leading spaces, but not trailing spaces.



    Args:

        text: The input text.



    Returns:

        The text with leading spaces removed from each line and trailing spaces trimmed.

    """

    lines: List[str] = text.splitlines()

    trimmed_lines: List[str] = []

    for line in lines:

        trimmed_line: str = line.lstrip()  # Remove leading spaces

        trimmed_line = trimmed_line.rstrip()  # Trim trailing spaces

        trimmed_lines.append(trimmed_line)

    return '\n'.join(trimmed_lines)

