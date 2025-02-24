from typing import Tuple



def replace_all_recursive(s: str, old: str, new: str) -> str:

    """Replaces all occurrences of the substring `old` with `new` in the string `s`.



    Args:

        s: The input string.

        old: The substring to be replaced.

        new: The substring to replace `old` with.



    Returns:

        The string `s` with all occurrences of `old` replaced with `new`.

    """

    if old not in s:

        return s

    start = s.index(old)

    length = len(old)

    return s[:start] + new + replace_all_recursive(s[start + length:], old, new)

