from typing import Tuple



def gamma_mean_and_std_to_shape_and_scale(mean: float, std: float) -> Tuple[float, float]:

    """Converts mean and standard deviation to shape and scale parameters for a Gamma distribution.



    Args:

        mean: The mean of the Gamma distribution.

        std: The standard deviation of the Gamma distribution.



    Returns:

        A tuple containing the shape and scale parameters.

    """

    shape = mean**2 / std**2

    scale = std**2 / mean

    return (shape, scale)

