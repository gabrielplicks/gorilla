import struct



def read_int64(filename: str, endianness: str) -> int:

    """Reads a 64-bit signed integer from a binary file with a given endianness.



    Args:

        filename: The name of the binary file.

        endianness: The endianness of the binary file, either "little" or "big".



    Returns:

        The 64-bit signed integer read from the binary file.



    Raises:

        ValueError: If the endianness is not specified as either "little" or "big".

    """

    with open(filename, "rb") as f:

        if endianness == "little":

            fmt = "<q"

        elif endianness == "big":

            fmt = ">q"

        else:

            raise ValueError("Invalid endianness")



        return struct.unpack(fmt, f.read(8))[0]

