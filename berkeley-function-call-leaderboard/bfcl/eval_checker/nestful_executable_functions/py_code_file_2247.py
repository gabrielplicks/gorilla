from typing import Dict



def extract_rate_limit_info(headers: Dict[str, str]) -> Dict[str, int]:

    """Extracts rate limit information from HTTP headers and converts it to the expected format.



    The format is a dictionary with `limit`, `remaining`, and `reset` keys.

    The `headers` dictionary is used as the source of headers.



    Args:

        headers: The headers dictionary.



    Returns:

        A dictionary with `limit`, `remaining`, and `reset` keys.

    """

    limit_info = {

        'limit': int(headers.get('rate_limit')),

        'remaining': int(headers.get('rate_limit_remaining')),

        'reset': int(headers.get('rate_limit_reset'))

    }

    return limit_info

