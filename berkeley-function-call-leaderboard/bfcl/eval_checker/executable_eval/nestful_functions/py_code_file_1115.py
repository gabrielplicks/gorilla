from typing import Tuple



def replace_phrase(file_path: str, old_phrase: str, new_phrase: str) -> str:

    """Reads the contents of a text file and replaces all occurrences of a given phrase.



    Args:

        file_path: The path to the text file.

        old_phrase: The phrase to be replaced.

        new_phrase: The phrase to replace the old one.



    Returns:

        The modified content as a string.

    """

    with open(file_path, "r") as file:

        content = file.read()

    modified_content = content.replace(old_phrase, new_phrase)

    return modified_content

