from typing import Union



def neutron_mass(temperature_in_kelvin: Union[int, float]) -> Union[float, str]:

    """Computes the mass of a neutron given an input of a non-negative temperature in Kelvin.

    Args:

        temperature_in_kelvin: The temperature in Kelvin.

    Raises:

        Exception: If the input temperature is less than zero.

    """

    if temperature_in_kelvin < 0:

        raise Exception("Input temperature cannot be less than zero.")

    elif temperature_in_kelvin >= 59.3:

        return 0.93827208816e9

    else:

        return 1.674927471 * 10**(-27) * (1 + temperature_in_kelvin / (8.617333262 * 10**12))**(5/3)

