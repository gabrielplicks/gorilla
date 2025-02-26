from typing import Union



def get_error_category(error_code: Union[int, str]) -> Union[str, None]:

    """Returns the category of a given error code.



    Args:

        error_code: The error code to get the category for.



    Returns:

        The category of the error code, or None if the error code is invalid.

    """

    if error_code == 000:

        return None

    elif 100 <= error_code <= 199:

        return "ConnectionError"

    elif 200 <= error_code <= 299:

        return "TimeoutError"

    elif 300 <= error_code <= 399:

        return "InvalidTokenError"

    elif error_code == 999:

        return "UnknownError"

    else:

        return None

