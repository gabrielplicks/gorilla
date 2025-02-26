from typing import Dict



def transform_text(dictionary: Dict[str, str], text: str) -> str:

    """Transforms a string by replacing each occurrence of a key with its corresponding value from the dictionary.



    Args:

        dictionary: A dictionary where keys are strings to be replaced and values are their replacements.

        text: The string to be transformed.

    """

    for key, value in dictionary.items():

        text = text.replace(key, value)



    return text

