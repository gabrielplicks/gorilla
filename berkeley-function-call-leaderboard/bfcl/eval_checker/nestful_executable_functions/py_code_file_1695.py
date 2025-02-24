from typing import Optional



def str_starts_with(string: str, prefix: Optional[str]) -> bool:

    """Checks if a string starts with a given prefix and returns True or False.

    If the prefix is a null string, the function returns True.

    Args:

        string: The string to check.

        prefix: The prefix to check for.

    """

    return string.startswith(prefix) if prefix else True

