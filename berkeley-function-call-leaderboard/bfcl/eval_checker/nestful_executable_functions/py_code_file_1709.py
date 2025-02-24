from typing import Set



def remove_aeiou(s: str) -> str:

    """Removes all characters in the input string that appear in the string "aeiouAEIOU".



    Args:

        s: The input string.



    Returns:

        A new string with all characters in `s` that appear in the string "aeiouAEIOU" removed.

        If the input string contains only characters in this list, the function returns an empty string.

    """

    valid_chars: Set[str] = set("aeiouAEIOU")

    return "".join(c for c in s if c not in valid_chars)

