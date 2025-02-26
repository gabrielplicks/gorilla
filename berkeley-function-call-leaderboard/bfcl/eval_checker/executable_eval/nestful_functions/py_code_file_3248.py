def convert_string_ascii(s: str) -> str:

    """Converts a string of lowercase English letters to a string of ASCII values.



    Args:

        s: A string of lowercase English letters.



    Returns:

        A string of ASCII values separated by hyphens.

    """

    if not s.isalpha() or not s.islower():

        return ''



    ascii_values = []



    for char in s:

        ascii_values.append(str(ord(char)))



    return '-'.join(ascii_values)

