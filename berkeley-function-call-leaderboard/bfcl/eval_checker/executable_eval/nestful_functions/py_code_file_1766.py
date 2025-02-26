from typing import List



def diff_eq_with_constant(n: int, x: List[float]) -> List[float]:

    """Computes the partial derivative of the dependent variable y with respect to the independent variable x.



    Args:

        n: An integer representing a standard partial differential equation (PDE) with a constant signal.

        x: A list of numbers representing the independent variable in the PDE.



    Returns:

        A list of numbers representing the dependent variable in the PDE.

    """

    y = []

    for i in range(len(x)):

        dy = n * x[i]

        y.append(dy)

    return y

