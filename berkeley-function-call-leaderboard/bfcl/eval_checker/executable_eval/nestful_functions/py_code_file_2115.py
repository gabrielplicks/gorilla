from typing import Dict



def check_uniqueness(dictionary: Dict[str, str]) -> Dict[str, str]:

    """Checks if all the values in the dictionary are unique.



    Args:

        dictionary: The dictionary to check.



    Raises:

        ValueError: If there are duplicates in the values.



    Returns:

        The input dictionary if there are no duplicates.

    """

    values = dictionary.values()

    if len(values) != len(set(values)):

        raise ValueError("Value Duplicates Found!")

    return dictionary

