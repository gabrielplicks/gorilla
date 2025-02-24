import re

from typing import Dict, Optional



def parse_currency_string(currency_string: str) -> Optional[Dict[str, str]]:

    """Converts a string containing currency information into a dictionary.

    Args:

        currency_string: The input string containing currency information.

    Returns:

        A dictionary with the following fields: `currency`, `amount`, and `type`.

        If the input string does not match the expected format, the function returns `None`.

    """

    pattern = r"^(?P<currency>\w{3})\s+(?P<amount>\d+)\s+(?P<type>sell|buy)$"

    match = re.match(pattern, currency_string)

    if match:

        parsed_data = match.groupdict()

        parsed_data["amount"] = int(parsed_data["amount"])

        return parsed_data

    return None

