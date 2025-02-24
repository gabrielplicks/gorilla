import re



def html_entities(text: str) -> str:

    """

    Generates HTML entities from a given string by replacing special characters with their corresponding HTML entities.

    Args:

        text: The input string to be converted to HTML entities.

    """

    special_chars = {

        '&': '&amp;',

        '<': '&lt;',

        '>': '&gt;',

        '"': '&quot;'

    }

    regex = re.compile('|'.join(map(re.escape, special_chars)))



    def replace_special_chars(match):

        return special_chars[match.group()]



    return regex.sub(replace_special_chars, text)

