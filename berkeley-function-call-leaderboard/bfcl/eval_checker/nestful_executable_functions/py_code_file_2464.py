from typing import Union



def calculate_force(mass: Union[int, float]) -> float:

    """Calculates the force acting on an object with a given mass.



    The function assumes a gravitational acceleration of 9.8 m/s^2 and returns the result in Newtons.



    Args:

        mass: The mass of the object.



    Returns:

        The force acting on the object in Newtons.

    """

    return mass * 9.8



def test_calculate_force():

    assert calculate_force(1) == 9.8

    assert calculate_force(2) == 19.6

    assert calculate_force(3) == 29.4

