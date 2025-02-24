from typing import Dict



def consecutive_character_counts(string: str) -> str:

    """Calculates the number of consecutive characters for each character in the input string.

    Args:

        string: The input string.

    Returns:

        A string containing the character and its count for each character in the input string.

    """

    character_counts: Dict[str, int] = {}



    for character in string:

        character_counts[character] = character_counts.get(character, 0) + 1



    result = []

    for character, count in character_counts.items():

        result.append(character + str(count))



    return ''.join(result)

