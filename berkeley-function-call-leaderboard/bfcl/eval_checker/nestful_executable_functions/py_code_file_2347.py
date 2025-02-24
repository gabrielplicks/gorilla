from typing import Dict, List



def has_required_attributes(user: Dict, required_attributes: List[str]) -> bool:

    """Checks if a user object has all the required attributes.



    Args:

        user: A user object with a `.traits` attribute.

        required_attributes: A list of required attributes.



    Returns:

        True if the user object has all the required attributes, False otherwise.

    """

    for attribute in required_attributes:

        if attribute not in user.traits or user.traits[attribute] is None:

            return False

    return True

