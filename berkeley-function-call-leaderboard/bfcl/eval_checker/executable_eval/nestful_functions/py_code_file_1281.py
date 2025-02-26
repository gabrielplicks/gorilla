from typing import Dict



character_base_attributes = {

    'strength': 10,

    'perception': 10,

    'endurance': 10,

    'charisma': 10,

    'intelligence': 10,

    'agility': 10,

    'luck': 10

}



def merge_chars_with_options(character_options: Dict[str, int]) -> Dict[str, int]:

    """Merges the attributes of the character_options with the built-in Fallout 76 character attributes.



    Args:

        character_options: A dictionary of character attributes to merge with the base attributes.



    Returns:

        A dictionary of merged character attributes.

    """

    merged_attributes = character_base_attributes.copy()

    merged_attributes.update(character_options)

    return merged_attributes

