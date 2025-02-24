from typing import List



def to_vars(strings: List[str], prefix: str = 'var') -> List[str]:

    """Converts a list of strings into a list of variable names.



    Args:

        strings: A list of strings.

        prefix: An optional prefix for the variable names. Defaults to 'var'.



    Returns:

        A list of formatted variable names, with the prefix and a number (starting at 1) appended to each string.

    """

    return [f'{prefix}{i+1}' for i, s in enumerate(strings)]

