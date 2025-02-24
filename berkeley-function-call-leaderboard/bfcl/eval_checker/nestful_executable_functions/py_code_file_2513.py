from typing import Union



def calculate_fuel_consumption(distance: Union[int, float], fuel_usage: Union[int, float]) -> float:

    """Calculates the fuel consumption per 100 km based on the given distance and fuel usage.



    Args:

        distance: The distance traveled, in kilometers.

        fuel_usage: The amount of fuel used, in liters.



    Returns:

        The fuel consumption per 100 km, in liters per 100 km.



    Raises:

        ValueError: If the distance or fuel usage is less than or equal to 0.

    """

    if distance <= 0 or fuel_usage <= 0:

        raise ValueError("Invalid distance or fuel usage value.")

    fuel_consumption = fuel_usage / distance * 100

    return round(fuel_consumption, 2)

