from typing import Tuple



def decode_icmp_echo_reply(data: bytes) -> Tuple[int, int, int, int, bytes]:

    """Decodes ICMP echo reply messages from the given binary data.



    Args:

        data: The binary data containing the ICMP packet.



    Returns:

        A tuple of `(type, code, checksum, ident, data)` where `data` is the

        corresponding bytes of the data section.

    """

    type = int.from_bytes(data[:1], byteorder='big')

    code = int.from_bytes(data[1:2], byteorder='big')

    checksum = int.from_bytes(data[2:4], byteorder='big')

    ident = int.from_bytes(data[4:8], byteorder='big')

    data = data[8:]



    return type, code, checksum, ident, data

