from typing import List



def convert_to_uppercase_string(original_string: str) -> str:

    """Converts a string to a new string where each character in the new string is the uppercase version of the previous character.



    Args:

        original_string: The original string to convert.



    Returns:

        The new string with each character in uppercase.

    """

    new_characters: List[str] = []

    for character in original_string:

        new_characters.append(character.upper())

    new_string: str = "".join(new_characters)

    return new_string

