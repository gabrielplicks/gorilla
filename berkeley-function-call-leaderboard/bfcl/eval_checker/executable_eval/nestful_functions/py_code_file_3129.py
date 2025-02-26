import re



def normalize_http_headers(header_name: str) -> str:

    """

    Normalizes HTTP headers by converting any header names to lowercase and replacing

    all hyphens with underscores. The function validates the input header name and

    returns a normalized version if it is valid. If the input is not a valid header name,

    the function raises an exception or returns a specific error message.



    Args:

        header_name: The HTTP header name to be normalized.



    Raises:

        ValueError: If the input header name is not valid.

    """

    if not re.match(r'^[a-zA-Z0-9-]+$', header_name):

        raise ValueError(f'Invalid header name: {header_name}')

    normalized_header_name = header_name.lower().replace('-', '_')

    return normalized_header_name

