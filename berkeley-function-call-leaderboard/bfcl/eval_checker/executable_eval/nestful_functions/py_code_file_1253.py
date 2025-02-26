from typing import Union



def convert_to_x(n: Union[int, float, str]) -> str:

    """Converts a number to its hexadecimal representation.

    Args:

        n: The number to be converted.

    """

    try:

        n_int = int(n)

    except ValueError:

        try:

            n_int = int(float(n))

        except ValueError:

            raise ValueError("n is not a valid input")



    if n_int < 0:

        return f"-{hex(abs(n_int))[2:]}"

    else:

        return hex(n_int)[2:]

