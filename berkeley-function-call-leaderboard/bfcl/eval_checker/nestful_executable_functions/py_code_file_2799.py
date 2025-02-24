from typing import Dict



def check_event(event: Dict[str, any]) -> None:

    """Checks if the dictionary has the following 3 keys: "type", "payload", and "user".

    If it does not, raise an exception with the message "Invalid event: missing type, payload, or user".



    Args:

        event: The dictionary to check.

    """

    required_keys = ["type", "payload", "user"]



    if not all(key in event for key in required_keys):

        raise Exception("Invalid event: missing type, payload, or user")

