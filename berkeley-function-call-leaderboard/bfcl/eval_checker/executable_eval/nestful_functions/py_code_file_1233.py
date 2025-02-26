from typing import Dict



def set_http_header(response: Dict, header_name: str, header_value: str) -> Dict:

    """Sets an HTTP header in a response.



    Args:

        response: A dictionary representing the HTTP response.

        header_name: The name of the HTTP header to set.

        header_value: The value of the HTTP header.



    Returns:

        The modified response.

    """

    if not isinstance(response, dict):

        raise TypeError('Response must be a dictionary')

    if not isinstance(header_name, str):

        raise TypeError('Header name must be a string')

    if not isinstance(header_value, str):

        raise TypeError('Header value must be a string')

    if 'headers' not in response:

        response['headers'] = {}

    if 'status_code' not in response:

        response['status_code'] = 200

    response['headers'][header_name] = header_value

    return response

