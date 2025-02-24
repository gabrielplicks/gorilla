import uuid

from typing import Union



def convert_uuid_string_to_object(uuid_string: str) -> Union[uuid.UUID, str]:

    """Converts a UUID string to a UUID object.



    Args:

        uuid_string: A 32-character hexadecimal UUID string.



    Returns:

        A UUID object if the input string is valid, or an error message if the input string is missing or malformed.

    """

    try:

        uuid_object = uuid.UUID(uuid_string)

    except (ValueError, AttributeError):

        return "Invalid UUID string. Must be a 32-character hexadecimal value."

    return uuid_object

