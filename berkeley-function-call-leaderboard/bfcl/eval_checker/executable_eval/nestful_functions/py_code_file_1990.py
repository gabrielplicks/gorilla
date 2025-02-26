from typing import List



def convert_string_to_list_of_bytes(string: str) -> List[str]:

    """Converts a string to a list of bytes in the format '0x11, 0x22, 0x33, ...'.



    Args:

        string: The input string.



    Returns:

        A list of bytes in the format '0x11, 0x22, 0x33, ...'.

    """

    byte_list = []

    bytes_obj = bytes(string, 'utf-8')

    for byte in bytes_obj:

        byte_list.append(f'0x{byte:02x}')

    return byte_list

