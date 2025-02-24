from typing import Optional



def classify_chinese_character(character: str) -> Optional[str]:

    """Classifies a Chinese character into one of the five elements: Wood, Fire, Earth, Metal, Water.



    Args:

        character: The Chinese character to classify.



    Returns:

        The corresponding element if the character is in the mapping, or None if it's not found.

    """

    element_mapping = {

        "木": "Wood",

        "火": "Fire",

        "土": "Earth",

        "金": "Metal",

        "水": "Water"

    }

    return element_mapping.get(character, None)

