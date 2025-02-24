from typing import Optional



def replace_title(text: str) -> Optional[str]:

    """Replaces certain titles in a given string.



    If the string contains the pattern "Mr. " or "Mrs. ", replace it with "Miss. ".

    If the string contains the pattern "Ms. ", replace it with "Mrs. ".



    Args:

        text: The input string.

    """

    text = text.replace("Mr. ", "Miss. ")

    text = text.replace("Mrs. ", "Miss. ")

    text = text.replace("Ms. ", "Mrs. ")

    return text

