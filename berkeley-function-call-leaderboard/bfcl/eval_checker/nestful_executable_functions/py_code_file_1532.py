from typing import Union



def printable_string_from_binary_file(file_path: str) -> str:

    """

    Converts a binary file into a string containing its printable characters.

    Non-printable characters are escaped appropriately.



    Args:

        file_path: The path to the binary file.



    Returns:

        A string containing the printable characters and escaped non-printable characters.

    """

    with open(file_path, "rb") as file:

        file_bytes = file.read()



    printable_string = ""



    for byte in file_bytes:

        if byte >= 32 and byte <= 126:

            printable_string += chr(byte)

        else:

            printable_string += f"\\x{byte:02x}"



    return printable_string

