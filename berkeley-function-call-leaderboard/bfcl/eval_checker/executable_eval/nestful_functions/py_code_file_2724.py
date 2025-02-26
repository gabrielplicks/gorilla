from typing import Tuple



def hex_to_float(hex_string: str) -> Tuple[int, int, int]:

    """Converts a hexadecimal string into its binary representation and extracts the sign bit, exponent, and mantissa bits.



    Args:

        hex_string: A hexadecimal string representing a single-precision floating-point number.



    Returns:

        A tuple containing the sign bit, exponent bits, and mantissa bits.

    """

    bin_string = format(int(hex_string, 16), "032b")

    sign_bit = int(bin_string[0])

    exponent_bits = int(bin_string[1:9], 2)

    mantissa_bits = int(bin_string[9:], 2)



    return (sign_bit, exponent_bits, mantissa_bits)

