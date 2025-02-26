from typing import Tuple



def complex_multiply(z1: Tuple[int, int], z2: Tuple[int, int]) -> Tuple[int, int]:

    """Multiplies two complex numbers without using the built-in `complex` type.



    Args:

        z1: A tuple representing the real and imaginary parts of the first complex number.

        z2: A tuple representing the real and imaginary parts of the second complex number.



    Returns:

        A tuple containing the real and imaginary parts of the product.

    """

    a, b = z1

    c, d = z2

    real = a * c - b * d

    imag = a * d + b * c

    return real, imag

