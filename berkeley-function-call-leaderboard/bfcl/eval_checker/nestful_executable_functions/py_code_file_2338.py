import textwrap



def add_space_to_str_lines(text: str) -> str:

    """Adds a space to each line of the input string while preserving the original indentation.



    Args:

        text: The input string.



    Returns:

        The modified string with a space added to each line.

    """

    lines = text.split("\n")

    wrapped_lines = [textwrap.fill(line, initial_indent=" ", subsequent_indent=" ") for line in lines]

    return "\n".join(wrapped_lines)

