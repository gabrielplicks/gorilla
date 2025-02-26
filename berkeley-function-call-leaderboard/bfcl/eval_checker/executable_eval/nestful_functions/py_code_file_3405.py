from typing import Tuple



def get_next_mac_address(address: str) -> str:

    """Gets the next valid MAC address in the format '02-42-AC-13-00-00' (lower case).



    Args:

        address: The input MAC address string.



    Returns:

        The next valid MAC address string.

    """

    parts = address.split('-')

    last_byte = int(parts[-1], 16)

    next_last_byte = (last_byte + 1) % 256  # Wrap around if the last byte reaches 256

    parts[-1] = f'{next_last_byte:02X}'

    return '-'.join(parts)

