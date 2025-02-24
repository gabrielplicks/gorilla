from typing import AnyStr



def validate_and_preprocess(text: AnyStr) -> str:

    """Validates and pre-processes user input.



    Args:

        text: The input string to be validated and pre-processed.

    """

    text = text.strip()

    text = "".join(filter(str.isalnum, text))

    text = text.lower()



    return text

