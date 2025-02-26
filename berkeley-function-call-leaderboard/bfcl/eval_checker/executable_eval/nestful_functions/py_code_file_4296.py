from typing import Union



def compute_hypotenuse(leg1: Union[int, float], leg2: Union[int, float]) -> Union[int, float]:

    """Computes the length of the hypotenuse of a right triangle given the lengths of the other two sides.



    Args:

        leg1: The length of the first leg of the right triangle.

        leg2: The length of the second leg of the right triangle.



    Returns:

        The length of the hypotenuse of the right triangle.

    """

    return (leg1 ** 2 + leg2 ** 2) ** 0.5

