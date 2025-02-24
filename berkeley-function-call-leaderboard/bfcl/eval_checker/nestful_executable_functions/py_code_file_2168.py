import re



def parse_md(markdown: str) -> str:

    """Parses a single Markdown header (H1) and returns it as a string.



    The header markup is composed of a pound symbol (#) followed by at least one space and a header text, which then ends with a newline.



    Args:

        markdown: The Markdown string to parse.



    Returns:

        The header text as a string, preserving the leading and trailing whitespace.



    Raises:

        ValueError: If the Markdown header format is invalid.

    """

    match = re.search(r'^# +(.*)\n$', markdown)

    if match:

        return match.group(1)

    else:

        raise ValueError("Invalid Markdown header format")

