import secrets

from typing import Dict



def get_random_file_name(attribute_dict: Dict[str, str]) -> str:

    """Generates a random file name based on the given attribute dictionary.



    The format of the file name is "[Random_Chars]_[value].txt", where [Random_Chars]

    represents a random sequence of 16 characters, and [value] is the value corresponding

    to the key "value" in the `attribute_dict`.



    Args:

        attribute_dict: A dictionary containing the attributes.



    Returns:

        A string representing the random file name.

    """

    if "value" not in attribute_dict:

        return None  # or return an error message

    value = attribute_dict["value"]

    random_chars = secrets.token_urlsafe(16)

    file_name = f"{random_chars}_{value}.txt"

    return file_name

