from typing import Dict, List



def character_ages(character_list: List[Dict[str, str]]) -> Dict[str, int]:

    """Extracts the names and ages of characters from a list of dictionaries.



    Args:

        character_list: A list of dictionaries, each corresponding to a different character in a story.

            Each dictionary should have the keys `name` and `age`, and the values should be a string

            and an integer, respectively.



    Returns:

        A dictionary called `characters`. The keys of this dictionary should be the `name`s,

        and the values should be the `age`s.

    """

    characters = {}

    for character in character_list:

        name = character['name']

        age = character['age']

        characters[name] = age

    return characters

