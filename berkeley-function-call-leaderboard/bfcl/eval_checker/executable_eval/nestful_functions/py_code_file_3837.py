from typing import Union



def get_initials(s: Union[str, None]) -> str:

    """Returns the initials of the words in the input string.



    Args:

        s: The input string.



    Returns:

        The initials of the words in the input string.

    """

    if not isinstance(s, str) or not s.strip():

        return ""



    initials = []

    for word in s.split():

        initials.append(word[0])



    return "".join(initials)

