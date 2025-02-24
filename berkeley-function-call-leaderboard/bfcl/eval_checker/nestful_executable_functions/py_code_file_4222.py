import logging

from typing import Any



def check_name_attribute(person: Any) -> bool:

    """Checks if a person object has a 'name' attribute.



    Args:

        person: The person object to check.



    Returns:

        True if the person has a 'name' attribute, False otherwise.

    """

    try:

        if hasattr(person, 'name'):

            return True

        else:

            logging.warning(f"Missing 'name' attribute for person {person}.")

            return False

    except Exception as e:

        logging.error(f"Error checking name attribute for person {person}: {e}")

        raise

