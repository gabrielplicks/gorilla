import re



def remove_formatting(text: str) -> str:

    """Removes formatting characters from a given string.



    Args:

        text: The input string.



    Returns:

        The input string without formatting characters and the new-line character at the end.

    """

    bold_pattern = re.compile(r'\[b\](.*?)\[/b\]')

    italic_pattern = re.compile(r'\[i\](.*?)\[/i\]')

    text = bold_pattern.sub(r'\1', text)

    text = italic_pattern.sub(r'\1', text)

    text = text.rstrip('\n')



    return text

