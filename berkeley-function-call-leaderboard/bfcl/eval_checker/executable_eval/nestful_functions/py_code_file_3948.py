from typing import Tuple



def pack_channels(r: int, g: int, b: int) -> int:

    """Packs the given red, green, and blue color channels into a single integer value.



    Args:

        r: The red color channel value.

        g: The green color channel value.

        b: The blue color channel value.



    Returns:

        A 32-bit integer value where the most significant 8 bits are the red channel,

        the next 8 bits are the green channel, and the last 8 bits are the blue channel.

    """

    return (r << 16) | (g << 8) | b

