import re

import typing



def camelCase_to_space_separated(s: str) -> str:

    """Converts a camelCase string to a space separated string.



    Args:

        s: The camelCase string to convert.



    Returns:

        The space separated string.

    """

    matches = re.findall('[A-Z]', s)

    result = re.sub('[A-Z]', ' \\g<0>', s)

    return result[0].lower() + result[1:]

