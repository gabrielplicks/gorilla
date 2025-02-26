from typing import Union



def first_alphabetical_word(s: str) -> Union[str, None]:

    """Returns the first word in the string if the string starts with an alphabetical character.

    If the string does not start with an alphabetical character, the function returns the second word in the string.

    Args:

        s: The input string.

    """

    s = s.strip()

    if s[0].isalpha():

        return s.split()[0]

    else:

        return s.split()[1]

