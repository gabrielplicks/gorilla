from typing import Union



def get_n_characters(string: str, n: int) -> Union[str, None]:

    """

    Returns a substring of the first `n` characters of the input string `string`.



    Args:

        string: The input string.

        n: The number of characters to return.



    Returns:

        A substring of the first `n` characters of `string`, or `string` itself if `n` is greater than the length of `string`.

        If `n` is less than 0, returns an empty string.

    """

    if n < 0:

        return ""

    elif n > len(string):

        return string

    else:

        return string[:n]

