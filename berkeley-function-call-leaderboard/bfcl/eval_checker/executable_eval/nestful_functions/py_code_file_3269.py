import uuid



def generate_uuid_from_string(string: str) -> str:

    """

    Generates a UUID from a string.



    Args:

        string: The string to be converted to a UUID.



    Raises:

        ValueError: If the string is empty.

    """

    if not string:

        raise ValueError('String must not be empty')



    return str(uuid.uuid5(uuid.NAMESPACE_URL, string))

