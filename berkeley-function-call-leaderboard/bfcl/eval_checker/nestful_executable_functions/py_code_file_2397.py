from typing import List



def parse_names(name_string: str) -> str:

    """Parses a string of comma-delimited names, returns a list of formatted names, and converts the list into a string of comma-separated names.

    Args:

        name_string: A string of comma-delimited names.

    Returns:

        A string of comma-separated names.

    """

    names = name_string.split(',')

    formatted_names: List[str] = []

    for name in names:

        name = name.strip()

        name = ' '.join(word.capitalize() for word in name.split())

        formatted_names.append(name)

    return ','.join(formatted_names)

