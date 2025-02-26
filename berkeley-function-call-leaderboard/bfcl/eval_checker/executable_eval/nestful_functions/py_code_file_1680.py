import math



def check_nan(arr: list) -> bool:

    """Checks if the given array has any NaN values.



    Args:

        arr: The input array to check for NaN values.



    Returns:

        True if the array has any NaN values, False otherwise.

    """

    for element in arr:

        if math.isnan(element):

            return True

    return False

