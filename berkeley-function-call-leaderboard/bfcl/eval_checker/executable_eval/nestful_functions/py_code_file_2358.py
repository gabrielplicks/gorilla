import re

from typing import Union



def camel_to_kebab(s: Union[str, bytes]) -> str:

    """Converts a string from camel case to kebab case.



    Args:

        s: The input string.



    Returns:

        The input string converted to kebab case.

    """

    pattern = r'(?<=[a-z0-9])[A-Z]|(?<!^)[A-Z](?=[a-z])'

    return re.sub(pattern, lambda m: '-' + m.group().lower(), s)

