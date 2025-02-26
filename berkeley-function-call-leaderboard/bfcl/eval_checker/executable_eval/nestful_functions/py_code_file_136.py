from typing import Dict



def get_cookie_value(headers: Dict[str, str], cookie_name: str) -> str:

    """Parses request headers to extract a specific cookie value by its name.



    Args:

        headers: A dictionary with keys representing header names and values representing their corresponding values.

        cookie_name: The name of the cookie to extract the value for.



    Returns:

        The value of the cookie if found, otherwise `None`.

    """

    cookie_value = None

    for key, value in headers.items():

        if key == 'Cookie':

            cookies = value.split(';')

            for cookie in cookies:

                parts = cookie.split('=')

                if parts[0].strip() == cookie_name:

                    cookie_value = parts[1].strip()

    return cookie_value

