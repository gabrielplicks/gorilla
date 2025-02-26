from typing import Dict



def double_dict(dictionary: Dict[str, float]) -> Dict[str, float]:

    """Creates a new dictionary with each value being the key's value multiplied by 2.

    Args:

        dictionary: The original dictionary.

    Returns:

        A new dictionary with updated values.

    """

    try:

        return {key: value * 2 for key, value in dictionary.items()}

    except Exception:

        return None

