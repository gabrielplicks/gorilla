import numpy as np



def activation_function(x: float) -> float:

    """Calculates the activation value of a neuron given its input.



    Args:

        x: The input to the neuron.



    Returns:

        The activation value of the neuron.

    """

    return np.maximum(0, x) + np.log(1 + np.exp(-abs(x)))

