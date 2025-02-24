from typing import Dict



def extract_device_id(payload: Dict[str, str], default_value: str) -> str:

    """Extracts the value of the "device_id" key from the payload dictionary.



    Args:

        payload: The payload dictionary.

        default_value: The default value to return if the "device_id" key is not present in the payload dictionary.



    Returns:

        The value of the "device_id" key from the payload dictionary, or the default value if the key is not present.

    """

    return payload.get("device_id", default_value)

