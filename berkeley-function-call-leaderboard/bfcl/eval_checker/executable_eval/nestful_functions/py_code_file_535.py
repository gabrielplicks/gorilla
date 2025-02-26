from typing import Union



def is_error_code(status_code: Union[int, str]) -> bool:

    """Checks if a given HTTP status code is an error code (4xx or 5xx).



    Args:

        status_code: The HTTP status code.



    Returns:

        True if the status code is an error code and False otherwise.

    """

    return str(status_code).startswith('4') or str(status_code).startswith('5')

