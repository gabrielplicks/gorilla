import re

import string



def camel_to_title(string: str) -> str:

    """Converts a string that is in camelCase format into Title Case.

    Args:

        string: The string to be converted.

    """

    return re.sub(r"([a-z])([A-Z])", r"\1 \2", string).title()

