from typing import Union



def frequency_to_wavelength(frequency: Union[int, float]) -> float:

    """Converts frequency in Hertz to wavelength in meters.

    Args:

        frequency: The frequency in Hertz.

    """

    c = 2.99792458e8

    wavelength = c / frequency

    return wavelength

