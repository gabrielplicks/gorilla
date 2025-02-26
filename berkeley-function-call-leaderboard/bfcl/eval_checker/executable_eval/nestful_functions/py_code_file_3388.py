import re



def pascal_to_kebab(string: str) -> str:

    """Converts a pascal case string to a kebab case string.



    Args:

        string: The input pascal case string.



    Returns:

        The input string converted to kebab case.

    """

    return re.sub(r"(?<!^)(?=[A-Z])", "-", string).lower()

