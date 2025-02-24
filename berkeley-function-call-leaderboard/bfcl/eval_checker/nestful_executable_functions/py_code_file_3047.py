import textwrap



def add_indent(text: str, level: int) -> str:

    """Adds indentation to a text block, and handles errors gracefully.

    Args:

        text: The text block to be indented.

        level: The number of times to indent the text.

    """

    try:

        indented_text = textwrap.indent(text, ' ' * 4 * level)

    except Exception:

        return text

    return indented_text

