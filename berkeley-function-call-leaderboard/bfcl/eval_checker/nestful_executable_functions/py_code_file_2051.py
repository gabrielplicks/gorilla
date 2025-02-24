from typing import Union



class HTTPRequest:

    def __init__(self, method: str, status_code: int):

        self.method = method

        self.status_code = status_code



def validate_http_request(request: Union[HTTPRequest, str]) -> bool:

    """Validates an HTTP request object.



    Args:

        request: The HTTP request object to validate.



    Returns:

        True if the request is a valid GET request with a status code of 200, False otherwise.

    """

    if not isinstance(request, HTTPRequest):

        return False

    if request.method != "GET":

        return False

    try:

        if request.status_code != 200:

            return False

    except Exception as e:

        print(f"An exception occurred: {e}")

        return False

    return True

