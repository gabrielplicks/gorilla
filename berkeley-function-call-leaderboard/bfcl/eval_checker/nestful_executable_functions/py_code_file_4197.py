from typing import List



def group_ip_addresses(ip_addresses: List[str]) -> List[List[str]]:

    """Groups a list of IP addresses into 8-bit blocks.



    Args:

        ip_addresses: A list of IP addresses.



    Returns:

        A list of 8-bit blocks, each block being a list of IP addresses.

        If an error occurs during the process, it returns None.

    """

    try:

        ip_blocks = []

        for ip_address in ip_addresses:

            blocks = ip_address.split('.')  # Split IP address into 8-bit blocks

            ip_blocks.append(blocks)

        return ip_blocks

    except Exception as e:

        print(f"Error: {e}")

        return None

