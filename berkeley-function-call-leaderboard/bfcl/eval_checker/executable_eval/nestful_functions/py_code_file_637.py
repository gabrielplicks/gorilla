from typing import Union



def extract_first_value(values: list[Union[str, None]]) -> Union[str, None]:

    """Extracts the first non-null value from a list of strings or values.



    Args:

        values: A list of strings or values.



    Returns:

        The first non-null value in the list, or None if all values are null.

    """

    for value in values:

        if value is not None:

            return value

    return None

