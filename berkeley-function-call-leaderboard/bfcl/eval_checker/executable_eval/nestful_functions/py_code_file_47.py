from typing import Optional



def is_name_valid(name: str) -> Optional[bool]:

    """Checks if the given name is valid using a tribool logic.



    A name is considered valid if it is composed of only alphabetic characters and has at least one vowel.

    If the name is valid, the function returns True, otherwise it returns False. If the name is invalid,

    the function returns None.



    Args:

        name: The name to check.



    Returns:

        True if the name is valid, False if the name is invalid, and None if the name contains non-alphabetic characters.

    """

    if not name.isalpha():

        return None



    vowels = {'a', 'e', 'i', 'o', 'u'}

    return any(letter in vowels for letter in name)

