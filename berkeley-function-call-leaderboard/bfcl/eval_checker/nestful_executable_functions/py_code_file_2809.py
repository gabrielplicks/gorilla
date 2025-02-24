import string



def check_valid_identifier(sequence: str) -> bool:

    """Checks if a given sequence of characters is a valid identifier in Python.



    Args:

        sequence: The sequence of characters to check.



    Returns:

        True if the sequence is a valid identifier, False otherwise.

    """

    return sequence.isidentifier() and sequence[0] in string.ascii_letters + '_'

