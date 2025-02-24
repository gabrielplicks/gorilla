from typing import Union



def extract_first_name(name: Union[str, None]) -> str:

    """Extracts the first name from a given string that contains the name in the format `first_name last_name`.

    Args:

        name: The input string containing the name.

    Returns:

        The first name as a string. If the string is empty or the format is not correct, returns an empty string.

    """

    try:

        words = name.split()

        return ' '.join(words[:words.index(words[-1])])

    except:

        return ''

