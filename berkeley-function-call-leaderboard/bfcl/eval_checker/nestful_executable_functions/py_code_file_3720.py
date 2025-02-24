from typing import List



def parse_http_response(http_response: str) -> List[str]:

    """Parses an HTTP response into a list of strings.



    Each string in the list is a single line of the response, including the status code and response headers.



    Args:

        http_response: A string of HTTP response data.



    Returns:

        A list of strings, where each string is a single line of the response.

    """

    lines = http_response.splitlines()

    output = []

    for line in lines:

        if line.startswith("HTTP"):

            output.append(line)

        else:

            output.append(line)

    return output

