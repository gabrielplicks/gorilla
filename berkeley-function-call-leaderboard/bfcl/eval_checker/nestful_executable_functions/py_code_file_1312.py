from typing import Dict



def validate_mapping(mapping: Dict[str, str]) -> bool:

    """Checks if the keys and values in a mapping are valid.



    Args:

        mapping: A dictionary with string keys and string values.



    Returns:

        True if the mapping is valid, False otherwise.

    """

    if not isinstance(mapping, dict):

        return False

    keys = set()

    values = set()

    for key, value in mapping.items():

        keys.add(key)

        values.add(value)

    if len(keys) != len(mapping) or len(values) != len(mapping.values()):

        return False

    if not all(isinstance(value, str) and len(value) > 0 for value in mapping.values()):

        return False

    return True

