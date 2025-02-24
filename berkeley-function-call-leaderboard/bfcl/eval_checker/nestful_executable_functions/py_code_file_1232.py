from typing import Tuple



def error_propagation(partial_x: float, partial_y: float, sigma_x: float, sigma_y: float) -> float:

    """Computes the standard deviation of a function given the partial derivatives and standard deviations of its variables.

    Args:

        partial_x: The partial derivative of the function with respect to x.

        partial_y: The partial derivative of the function with respect to y.

        sigma_x: The standard deviation of x.

        sigma_y: The standard deviation of y.

    """

    return (partial_x**2 * sigma_x**2 + partial_y**2 * sigma_y**2)**0.5

