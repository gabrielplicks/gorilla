from typing import Union



def complex_mod(number: Union[complex, float, int]) -> Union[float, None]:

    """Calculates the modulus of a complex number.



    Args:

        number: The complex number to calculate the modulus of.



    Returns:

        The modulus of the complex number as a float, or None if the input is not a complex number.

    """

    if isinstance(number, complex):

        return float(abs(complex(number)))

    return None

