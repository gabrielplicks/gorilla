import urllib.parse



def check_location_header(headers: list[tuple[str, str]]) -> bool:

    """Checks if there is a 'Location' header in a list of headers and if its value is a valid URL.



    Args:

        headers: A list of headers represented as tuples of header name and value.



    Returns:

        A boolean representing the match.

    """

    return any(header[0].lower() == 'location' and urllib.parse.urlparse(header[1]).scheme for header in headers)

