import math

from typing import Union



def complex_exponentiation(z: complex, n: int) -> complex:

    """Computes the exponentiation of a complex number z to the power of n, where n is a positive integer.

    Employs the built-in functions math.cos() and math.sinh() for complex number handling.

    Utilizes list comprehension to handle negative powers.



    Args:

        z: The complex number to be exponentiated.

        n: The power to which the complex number is to be raised.

    """

    r = abs(z)

    theta = math.acos(z.real / r)

    if n >= 0:

        return r ** n * (math.cos(n * theta) + math.sinh(n * theta) * 1j)

    else:

        return 1 / complex_exponentiation(z, -n)

