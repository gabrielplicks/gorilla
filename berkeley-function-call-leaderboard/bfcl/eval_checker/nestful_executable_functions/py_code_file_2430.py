import re



def convert_camel_case_to_snake_case(text: str) -> str:

    """

    Converts text in camelCase format to snake_case.



    Args:

        text: The text to convert.



    Returns:

        The converted text.

    """

    text = re.sub("([a-z0-9])([A-Z])", r"\1_\2", text).lower()

    return text

