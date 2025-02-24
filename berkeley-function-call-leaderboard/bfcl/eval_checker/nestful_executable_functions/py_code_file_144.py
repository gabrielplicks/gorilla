from typing import Union



def convert_to_c_style_identifier(name: str) -> str:

    """Converts a string `name` from a Python-style identifier to a C-style identifier.



    If the string does not start with an alphabetical character or underscore, prepend an underscore.

    If the string contains any uppercase characters, convert them to lowercase.

    If the string contains a hyphen, replace it with an underscore.

    If any other characters are encountered, ignore them.



    Args:

        name: The input string to be converted.



    Returns:

        The converted string.

    """

    if not name[0].isalpha() and name[0] != '_':

        name = '_' + name



    name = ''.join(c.lower() if c.isupper() else c for c in name)

    name = name.replace('-', '_')



    return ''.join(c for c in name if c.isalnum() or c == '_')

