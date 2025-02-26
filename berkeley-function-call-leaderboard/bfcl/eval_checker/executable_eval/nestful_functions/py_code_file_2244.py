import re

from typing import List



valid_ids: List[str] = ['user1', 'user_2', 'another_user']



def is_valid_id(id: str) -> bool:

    """Checks if a given string is a valid ID.



    A valid ID is a string of at least 2 characters, consisting only of alphanumeric

    characters and underscores, and cannot start with a number. The ID must also be

    present in a predefined list of valid IDs.



    Args:

        id: The string to check for validity.



    Returns:

        True if the string is a valid ID, False otherwise.

    """

    if len(id) < 2:

        return False

    if re.match(r'\d', id):

        return False

    if not re.match(r'^[a-zA-Z0-9_]+$', id):

        return False

    if id not in valid_ids:

        return False

    return True

