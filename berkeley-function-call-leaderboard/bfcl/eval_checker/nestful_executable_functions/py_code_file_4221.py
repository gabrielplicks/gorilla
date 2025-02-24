from typing import Dict



def escape_html(text: str) -> str:

    """Escapes a string for display in HTML.



    This function replaces special characters with their corresponding escape sequences.

    It is case-sensitive when escaping the characters.



    Args:

        text: The input string to be escaped.



    Returns:

        The escaped string.

    """

    escape_dict: Dict[str, str] = {

        "&": "&amp;",

        "<": "&lt;",

        ">": "&gt;",

        '"': "&quot;",

        "'": "&#39;",

    }



    escaped_text: str = ""

    for char in text:

        if char in escape_dict:

            escaped_text += escape_dict[char]

        else:

            escaped_text += char



    return escaped_text

