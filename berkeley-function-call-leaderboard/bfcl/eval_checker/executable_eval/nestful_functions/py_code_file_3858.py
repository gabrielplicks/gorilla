import re



def validate_alphanumeric_input(input_string: str) -> bool:

    """Validate that a string contains only alphanumeric characters.



    Args:

        input_string: The input string to validate.



    Returns:

        A boolean result indicating whether the input is valid or not.



    """

    pattern = r'^[a-zA-Z0-9]+$'  # Match alphanumeric characters only

    return bool(re.fullmatch(pattern, input_string))

