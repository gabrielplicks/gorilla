from typing import Optional



def replace_in_quotes(string: str, text: str, replacement_text: Optional[str] = None) -> str:

    """Replaces certain text in a string with a given replacement string, but only if the text is surrounded by quotation marks.



    Args:

        string: The original string.

        text: The text to find.

        replacement_text: The replacement text. If not provided, the original text will be returned.

    """

    if string.startswith('"') and string.endswith('"'):

        return string.replace(text, replacement_text)

    else:

        return string

