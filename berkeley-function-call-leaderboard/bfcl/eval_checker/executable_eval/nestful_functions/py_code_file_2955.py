from typing import Dict



def replace_abbreviations(text: str, abbreviations: Dict[str, str]) -> str:

    """Replaces a given list of abbreviations with their meanings in a text string.



    Args:

        text: The input string to modify.

        abbreviations: A dictionary of abbreviations and their meanings.



    Returns:

        The modified input string with the abbreviations replaced.

    """

    for abbreviation, meaning in abbreviations.items():

        if abbreviation in text:

            text = text.replace(abbreviation, meaning)

    return text

