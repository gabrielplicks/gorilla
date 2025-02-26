from typing import Union



def extract_environment(string: str) -> Union[str, None]:

    """Extracts the environment name from a given string.



    The environment name is a suffix designated by a marker, which is a dot (.)

    followed by a non-whitespace character. If the marker is not found, the

    function returns the given string without modification.



    Args:

        string: The input string.



    Returns:

        The environment name if the marker is found, otherwise the input string.

    """

    marker = '.'

    if marker in string:

        marker_index = string.index(marker)

        environment_name = string[marker_index + 1:].strip()

        return environment_name

    else:

        return string

