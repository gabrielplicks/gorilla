from typing import Union



def convert_to_mmhg(pressure_in_psi: Union[int, float]) -> float:

    """Converts a given pressure value in psi (pounds per square inch) to millimeters of mercury (mmHg).

    Args:

        pressure_in_psi: The pressure value in psi.

    """

    pressure_in_mmhg = round(pressure_in_psi * 51.715, 1)

    return pressure_in_mmhg

