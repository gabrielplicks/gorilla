from typing import Dict



def format_as_yaml(d: Dict[str, str]) -> str:

    """Formats a Python dictionary as a string in YAML format.



    Args:

        d: The input dictionary.



    Returns:

        A string in YAML format.

    """

    if not d:

        return ''



    s = ''

    for key, value in d.items():

        s += f'{key}: {value}\n'



    return s

