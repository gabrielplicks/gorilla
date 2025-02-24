from typing import Dict, Any



def key_exists(dictionary: Dict[str, Any], key: str) -> bool:

    """Checks if a key exists in a dictionary.

    Args:

        dictionary: The dictionary to check.

        key: The key to check for.

    """

    return key in dictionary

