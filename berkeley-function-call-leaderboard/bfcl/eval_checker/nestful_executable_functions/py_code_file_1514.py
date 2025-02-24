import re



def parse_http_header(header_line: str) -> dict:

    """Parses an HTTP header line and returns a dictionary containing the key and value as key-value pairs.



    Args:

        header_line: The HTTP header line to parse.



    Returns:

        A dictionary containing the key and value as key-value pairs.



    Raises:

        ValueError: If the header line is not valid.

    """

    key, value = header_line.split(':')

    key = key.strip()

    value = value.strip()

    if re.fullmatch(r'^[a-zA-Z0-9-]+$', key) and re.search(r'\S+', value):

        return {key: value}

    else:

        raise ValueError("Invalid header line")

