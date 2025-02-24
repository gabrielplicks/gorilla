from typing import Optional



def first_difference(a: str, b: str, ignore_case: Optional[bool] = False) -> int:

    """

    Returns the first index at which two strings start to differ, or -1 if they are identical.

    If `ignore_case` is True, performs a case-insensitive comparison.



    Args:

        a: The first string.

        b: The second string.

        ignore_case: If True, performs a case-insensitive comparison.

    """

    if ignore_case:

        a = a.lower()

        b = b.lower()

    for i in range(min(len(a), len(b))):

        if a[i] != b[i]:

            return i

    return -1

