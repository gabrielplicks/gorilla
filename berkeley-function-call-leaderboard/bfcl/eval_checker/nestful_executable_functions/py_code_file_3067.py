from typing import AnyStr



def check_string_starts_with_pattern(string: AnyStr, pattern: AnyStr) -> bool:

    """Checks if a string starts with a certain pattern.



    Args:

        string: The string to check.

        pattern: The pattern to check against.

    """

    return string.startswith(pattern)

