import re



def camel_case_to_snake_case(text: str) -> str:

    """Converts a string of CamelCase words to snake_case words.



    Args:

        text: The string of CamelCase words to convert.



    Returns:

        The string of snake_case words.

    """

    pattern = r'(?<!^)(?=[A-Z])'

    result = re.sub(pattern, '_', text)

    result = result.strip('_')

    return result.lower()

