import re

from typing import Union



def percentage_to_fraction(percentage_string: str) -> Union[float, None]:

    """Converts a string representing a percentage to a floating point number.



    Args:

        percentage_string: A string representing a percentage.



    Returns:

        A floating point number representing the percentage as a fraction between 0 and 1.



    Raises:

        ValueError: If the string is not a valid percentage.

    """

    pattern = r'^([0-9]{1,2}|100)%$'

    match = re.match(pattern, percentage_string)

    if match:

        number = int(match.group(1))

        fraction = number / 100

        return fraction

    else:

        raise ValueError("Invalid percentage string")

