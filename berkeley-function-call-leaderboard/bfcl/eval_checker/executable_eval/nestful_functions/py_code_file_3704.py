from typing import Any



def is_message_of_type(message: Any, type_str: str) -> bool:

    """Checks whether a given message is of a specific type.



    Args:

        message: The message object to check.

        type_str: The type string to check against.



    Returns:

        True if the message is of the given type, False otherwise.

    """

    if not hasattr(message, 'type'):

        return False

    return message.type == type_str

