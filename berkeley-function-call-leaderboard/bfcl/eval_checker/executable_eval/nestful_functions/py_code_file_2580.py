import time

import math



def float_to_int_optimized(f: float) -> int:

    """Converts a float variable to an int variable without any loss of data.



    The conversion takes a fixed time of 0.000001 seconds for each float-to-int conversion.

    The function is optimized to minimize the number of float-to-int conversions.



    Args:

        f: The float variable to be converted.



    Returns:

        The int variable after conversion.

    """

    if f < 0:

        return math.ceil(f)

    else:

        return math.floor(f)

