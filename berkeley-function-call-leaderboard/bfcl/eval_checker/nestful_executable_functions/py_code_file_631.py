from typing import AnyStr



def match_string_partially(s: AnyStr, sub: AnyStr) -> int:

    """

    Returns the index of the first occurrence of the substring `sub` in the string `s`, or `-1` if `sub` is not found.

    The function handles partial matches and does not rely on the built-in `.find()` method.



    Args:

        s: The string to search in.

        sub: The substring to search for.

    """

    if len(sub) > len(s):

        return -1



    for i in range(len(s) - len(sub) + 1):

        for j in range(len(sub)):

            if s[i+j] != sub[j]:

                break

        else:

            return i

    return -1

