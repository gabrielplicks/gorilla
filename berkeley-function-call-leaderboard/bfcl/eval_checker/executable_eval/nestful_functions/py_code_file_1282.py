import codecs



def utf8_to_binary(string: str) -> list[str]:

    """Converts a string into a list of binary strings representing the UTF-8 encoded bytes.



    Args:

        string: The input string.



    Returns:

        A list of binary strings, where each string represents a byte from the UTF-8 encoded string.

    """

    encoded_bytes = codecs.encode(string, "utf-8")

    return [format(byte, "b").zfill(8) for byte in encoded_bytes]

