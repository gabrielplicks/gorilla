import json

from typing import Dict



def format_http_request_header(method: str, path: str, host: str, headers: Dict[str, str] = None, token: str = None) -> str:

    """Formats an HTTP request header with a given method, path, host, and optional headers.

    Args:

        method: The HTTP method.

        path: The request path.

        host: The host.

        headers: A dictionary of optional headers.

        token: The token.

    Returns:

        The formatted HTTP request header.

    """

    if token is None:

        return "Token is missing"

    header = {

        "method": method,

        "path": path,

        "host": host,

        "token": token

    }

    if headers is not None:

        header.update(headers)

    return json.dumps(header, indent=2) + '\n'

