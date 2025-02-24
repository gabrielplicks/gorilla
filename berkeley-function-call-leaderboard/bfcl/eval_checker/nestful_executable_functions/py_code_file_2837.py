from typing import List



def camel_case(input_str: str) -> str:

    """Converts a string to camel case format.

    Args:

        input_str: The input string to be converted to camel case.

    """

    words: List[str] = input_str.split()

    camel_cased_words: List[str] = [words[0].lower()] + [word.title() for word in words[1:]]

    return "".join(camel_cased_words)

