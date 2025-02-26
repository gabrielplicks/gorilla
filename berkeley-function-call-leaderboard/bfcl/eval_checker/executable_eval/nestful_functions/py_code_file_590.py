import re



def validate_passport_id(passport_id: str) -> bool:

    """Validates a passport ID in the form of a string that is 9 characters long, starts with a letter, and contains only letters and digits.



    Args:

        passport_id: The passport ID to validate.



    Returns:

        True if the passport ID is valid, False otherwise.

    """

    pattern = r'^[a-zA-Z]\w{8}$'

    return bool(re.match(pattern, passport_id))

