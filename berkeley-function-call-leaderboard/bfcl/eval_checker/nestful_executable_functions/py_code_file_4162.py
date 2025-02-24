import numpy as np



def convert_dbz_to_rain_rate(reflectivity_dbz: np.ndarray) -> np.ndarray:

    """Converts a numpy array of reflectivity values in dBZ to rain rate in mm/hr.



    Args:

        reflectivity_dbz: A numpy array of reflectivity values in dBZ.



    Returns:

        A numpy array of rain rate values in mm/hr.

    """

    reflectivity_rain_rate = (reflectivity_dbz / 10.) ** 1.6

    return reflectivity_rain_rate

