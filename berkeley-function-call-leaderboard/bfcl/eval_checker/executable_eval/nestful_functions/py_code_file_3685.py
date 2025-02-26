from typing import Set



def format_and_remove_special_characters(text: str) -> str:

    """Formats and removes special characters from a given string.



    Args:

        text: The input string.



    Returns:

        The formatted string with special characters removed.

    """

    special_characters = set('!@#$%^&*()-_=+{}[]\|;:"\',<>/?.~`')



    formatted_text = []

    for i, character in enumerate(text):

        if character not in special_characters and character.isalnum():

            formatted_text.append(character)

        elif character in special_characters:

            if i == 0 or i == len(text) - 1:

                continue

            else:

                formatted_text.append('_')



    return ''.join(formatted_text)

