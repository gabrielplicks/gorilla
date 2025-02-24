from typing import List



def remove_vowels_ascii(text: str) -> str:

    """Removes the vowels from a string using the ASCII value of each character.



    Args:

        text: The input string.



    Returns:

        A new string that contains consonants only. If the string is empty or

        contains only vowels, returns an empty string.

    """

    vowel_ascii_codes: List[int] = [ord('a'), ord('e'), ord('i'), ord('o'), ord('u')]



    result: str = ""

    for c in text:

        result += c if ord(c) not in vowel_ascii_codes else ""



    return result

