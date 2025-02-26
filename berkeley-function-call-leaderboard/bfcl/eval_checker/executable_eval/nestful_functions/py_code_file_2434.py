from typing import Dict



def get_format_string(mapping: Dict[str, str]) -> str:

    """Constructs a format string with %s as placeholders for the values.



    Args:

        mapping: A dictionary of string-to-string mappings.



    Returns:

        A format string with %s as placeholders for the values.

    """

    placeholders = []

    for key, value in mapping.items():

        placeholders.append(f'{key} (%s)')

    return ', '.join(placeholders)

