from typing import Union



def change_characters(text: str, ch1: str, ch2: str) -> Union[str, None]:

    """Replaces all occurrences of ch1 in text with ch2 and returns the resulting string.

    Args:

        text: A string.

        ch1: A character.

        ch2: A character.

    """

    if not isinstance(text, str):

        return None



    result = ""

    for char in text:

        if char == ch1:

            result += ch2

        else:

            result += char



    return result

