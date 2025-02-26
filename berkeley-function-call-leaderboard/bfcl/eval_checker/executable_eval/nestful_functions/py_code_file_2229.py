from typing import Optional



def extract_user_agent_header(request_string: str) -> Optional[str]:

    """Extracts the value of the `User-Agent` header from an HTTP request string.



    Args:

        request_string: The HTTP request string.



    Returns:

        The value of the `User-Agent` header, or `None` if the header is not present.

    """

    lines = request_string.split("\n")



    for i, line in enumerate(lines):

        if line.startswith("User-Agent:"):

            header_value = line.split(":", 1)[-1].strip()

            return header_value



    return None

