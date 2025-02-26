from typing import Optional



def has_vowel(text: Optional[str]) -> bool:

    """Checks if a string contains at least one vowel.



    Args:

        text: The input string.



    Returns:

        True if the string contains at least one vowel, False otherwise.

    """

    if not text:

        return False

    vowels = {'a', 'e', 'i', 'o', 'u'}

    for c in text:

        if c.lower() in vowels:

            return True

    return False

