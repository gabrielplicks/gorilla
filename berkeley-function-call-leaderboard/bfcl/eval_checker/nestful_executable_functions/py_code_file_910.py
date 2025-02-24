from typing import List



def detect_suffix(s: str, suffixes: List[str]) -> bool:

    """Determines if the string `s` ends with one of the specified suffixes.



    Args:

        s: The string to check.

        suffixes: A list of suffixes to check against.



    Returns:

        A boolean value indicating whether `s` ends with one of the suffixes in the `suffixes` list.

    """

    return any(s.endswith(suffix) for suffix in suffixes)

