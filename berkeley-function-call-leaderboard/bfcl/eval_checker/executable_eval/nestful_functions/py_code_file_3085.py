from typing import AnyStr



def substring_check(string1: AnyStr, string2: AnyStr) -> bool:

    """

    Checks whether `string1` is a substring of `string2` regardless of case and

    handling of special characters.

    Args:

        string1: The first string to be checked.

        string2: The second string to be checked.

    """

    string1 = string1.lower().replace(" ", "").replace(",", "")

    string2 = string2.lower().replace(" ", "").replace(",", "")



    return string1 in string2

