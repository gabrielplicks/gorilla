from typing import Dict



def invert_unique_dict(d: Dict[str, int]) -> Dict[int, str]:

    """Inverts a dictionary where the values are unique.



    Args:

        d: A dictionary where values are unique.



    Returns:

        A dictionary where keys are values and values are keys.

    """

    return {value: key for key, value in d.items()}

