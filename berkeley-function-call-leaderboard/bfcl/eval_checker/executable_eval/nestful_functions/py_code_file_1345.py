from typing import Union



def calculate_kp(plasma_pressure: Union[int, float], plasma_density: Union[int, float]) -> float:

    """Calculates the plasma parameter $K_p$ for the simulation cell.

    Args:

        plasma_pressure: The plasma pressure.

        plasma_density: The plasma density.

    """

    return plasma_pressure / plasma_density

