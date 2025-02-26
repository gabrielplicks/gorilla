import re



def camel_to_snake_case(camel_case_word: str) -> str:

    """Converts a camelCased word to snake_case.



    Args:

        camel_case_word: The camelCased word to convert.



    Returns:

        The snake_case version of the input word.

    """

    snake_case_word = re.sub(r'([A-Z])', r'_\1', camel_case_word).lower()

    return snake_case_word

