import urllib.parse



def get_param_from_uri(uri: str, param: str) -> str:

    """Retrieves the value of a specified parameter from a URI string.



    Args:

        uri: The URI string.

        param: The parameter name.



    Returns:

        The parameter value as a string if it exists, or None if the parameter does not exist or is empty.

    """

    parsed_uri = urllib.parse.urlparse(uri)

    query_params = urllib.parse.parse_qs(parsed_uri.query)

    if param in query_params:

        return query_params[param][0]

    else:

        return None

