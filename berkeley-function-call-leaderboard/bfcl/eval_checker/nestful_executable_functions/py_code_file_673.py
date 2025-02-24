def calculate_average_speed(distance: float, time: float) -> float:

    """

    Calculates the average speed of a car based on the provided distance (km) and time (hrs).

    Args:

        distance: The distance traveled by the car in kilometers.

        time: The time taken to travel the distance in hours.

    """

    if time < 0:

        raise ValueError("Invalid time: cannot be negative.")



    speed = distance / time



    return round(speed, 1)

