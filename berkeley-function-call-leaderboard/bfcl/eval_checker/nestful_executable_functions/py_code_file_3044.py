import re

from typing import Optional, Tuple



def parse_result(input_string: str) -> Optional[Tuple[float, float, float]]:

    """Parses a string input representing the x-, y-, and z-coordinates of a point.



    Args:

        input_string: A string input representing the x-, y-, and z-coordinates of a point.



    Returns:

        The x-, y-, and z-coordinates as floating-point numbers if the input is in the correct format,

        otherwise None.

    """

    match = re.match(r"(\d+\.\d+) (\d+\.\d+) (\d+\.\d+)", input_string)

    if match:

        return (float(match.group(1)), float(match.group(2)), float(match.group(3)))

    return None

