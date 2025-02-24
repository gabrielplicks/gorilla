from typing import List



def validate_ip(ip_str: str) -> bool:

    """Validates if a string represents a valid IP address.



    Args:

        ip_str: The string to validate.



    Returns:

        True if the input represents a valid IP address, and False otherwise.

    """

    try:

        parts: List[str] = ip_str.split('.')

        if len(parts) != 4:

            return False

        for part in parts:

            if not part.isdecimal():  # Check if part is an integer

                return False

            if not 0 <= int(part) <= 255:

                return False

        return True

    except ValueError:

        return False

