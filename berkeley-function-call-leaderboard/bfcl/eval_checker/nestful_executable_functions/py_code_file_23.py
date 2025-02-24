from typing import Dict



def divide_distribution(distribution: Dict[float, float], threshold: float) -> tuple:

    """Divides a univariate probability distribution into two conditional distributions based on a given threshold.



    Args:

        distribution: A dictionary mapping values to their probabilities.

        threshold: The value that separates the two conditional distributions.



    Returns:

        A tuple of two dictionaries, where the first is the probability distribution for values below the threshold,

        and the second is the probability distribution for values above the threshold.

    """

    below_threshold = {}

    above_threshold = {}



    for key, value in distribution.items():

        if key < threshold:

            below_threshold[key] = value

        else:

            above_threshold[key] = value



    return below_threshold, above_threshold

