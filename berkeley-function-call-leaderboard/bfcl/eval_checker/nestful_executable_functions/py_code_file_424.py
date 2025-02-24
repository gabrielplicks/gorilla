from typing import Union



def snake_to_dash(string: Union[str, bytes]) -> Union[str, bytes]:

    """Converts a string from snake case to dash case.



    Args:

        string: The input string in snake case.



    Returns:

        The input string converted to dash case.

    """

    return string.replace('_', '-')

