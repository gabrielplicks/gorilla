from typing import List



def is_valid_header(http_header: List[object]) -> bool:

    """Checks if a given list of integers is a valid HTTP header.



    A valid header begins with a row of two integers that signify the number of key-value pairs in the header and the maximum key length.



    Args:

        http_header: A list of integers representing an HTTP header.



    Returns:

        True if the given list of integers is a valid HTTP header, False otherwise.

    """

    if not isinstance(http_header[0], int) or not isinstance(http_header[1], int) or http_header[0] < 0 or http_header[1] < 0:

        return False

    if not isinstance(http_header[2], str):

        return False

    if len(http_header[2]) != http_header[1]:

        return False



    return True

