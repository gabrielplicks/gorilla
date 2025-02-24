from typing import AnyStr



def snake_case_to_camel_case(snake_case_str: AnyStr) -> AnyStr:

    """Converts a snake_case string into a camelCase string.



    Args:

        snake_case_str: The snake_case string to be converted.



    Returns:

        The camelCase string.

    """

    camel_case_words = snake_case_str.replace("_", " ").title().split()

    camel_case_str = "".join(camel_case_words).lstrip("_")

    return camel_case_str

