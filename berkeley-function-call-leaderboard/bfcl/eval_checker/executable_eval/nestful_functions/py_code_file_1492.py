from typing import Union



def calculate_capacitor_energy(capacitance: Union[int, float], voltage: Union[int, float]) -> float:

    """Calculates the energy of a capacitor given its capacitance and voltage.



    Args:

        capacitance: The capacitance of the capacitor in farads.

        voltage: The voltage across the capacitor in volts.



    Returns:

        The energy of the capacitor in joules.

    """

    energy = 0.5 * capacitance * voltage ** 2

    return energy

