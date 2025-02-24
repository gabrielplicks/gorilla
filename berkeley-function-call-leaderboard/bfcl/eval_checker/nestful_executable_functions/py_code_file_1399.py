def alternate_concat(str1: str, str2: str) -> str:

    """Concatenates two strings by alternating their characters.



    If one string is longer than the other, append the remaining characters of the longer string to the end of the result.



    Args:

        str1: The first string.

        str2: The second string.



    Returns:

        The concatenated string.

    """

    result = ""

    for c1, c2 in zip(str1, str2):

        result += c1 + c2



    if len(str1) > len(str2):

        result += str1[len(str2):]

    elif len(str2) > len(str1):

        result += str2[len(str1):]



    return result

