import re



def validate_identifier(name: str) -> bool:

    """Validates if a string is a valid Python identifier.



    Args:

        name: The string to validate.



    Returns:

        True if the string is a valid Python identifier, False otherwise.

    """

    name = name.lower()



    if name[0].isdigit():

        return False



    if re.search(r'\W', name):

        return False



    return True

