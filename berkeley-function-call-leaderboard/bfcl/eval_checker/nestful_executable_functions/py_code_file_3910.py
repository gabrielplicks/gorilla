from typing import List



def values_in_frequency_domain(frequencies: List[float], sample_rate: float) -> List[float]:

    """Calculates the corresponding values in the frequency domain for a list of frequencies and a sample rate.



    Args:

        frequencies: A list of frequencies.

        sample_rate: The sample rate.



    Returns:

        A list of the corresponding values in the frequency domain.

    """

    return [2 * f / sample_rate for f in frequencies]

