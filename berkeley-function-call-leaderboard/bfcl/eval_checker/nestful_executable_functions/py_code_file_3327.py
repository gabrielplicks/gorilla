from typing import List



def create_string_list(values: List[float]) -> List[str]:

    """Creates a list of strings from a given list of integers or floats.



    Args:

        values: A list of integers or floats.



    Returns:

        A list of strings.

    """

    return [str(x) for x in values]

