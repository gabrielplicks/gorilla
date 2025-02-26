import re



def camel_case_to_underscore(s: str) -> str:

    """Converts a camel case string into an underscored string.



    Args:

        s: The camel case string to convert.



    Returns:

        The underscored string.

    """

    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)

    return s.lower()

