from typing import Dict



def adjust_weights(weights: Dict[str, float], multiplier: float) -> Dict[str, float]:

    """Adjusts the weights in a dictionary by multiplying each weight by a multiplier and adding 1.



    Args:

        weights: A dictionary of weights.

        multiplier: A scalar multiplier.



    Returns:

        A new dictionary with the adjusted weights.

    """

    adjusted_weights = {key: value * multiplier + 1 for key, value in weights.items()}

    return adjusted_weights

