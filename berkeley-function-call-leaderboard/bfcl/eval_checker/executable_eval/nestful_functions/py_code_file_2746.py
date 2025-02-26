from typing import Union



def get_meaningful_response_msg(resp_code: Union[int, str]) -> str:

    """Returns a meaningful text if the response code is within the 200-299 range, or raises an exception otherwise.



    Args:

        resp_code: An integer representing an HTTP response code.



    Raises:

        ValueError: If the response code is not within the 200-299 range.

    """

    if 200 <= resp_code <= 299:

        if resp_code == 200:

            return "OK"

        elif resp_code == 201:

            return "Created"

        elif resp_code == 202:

            return "Accepted"

    else:

        raise ValueError("Invalid response code")

