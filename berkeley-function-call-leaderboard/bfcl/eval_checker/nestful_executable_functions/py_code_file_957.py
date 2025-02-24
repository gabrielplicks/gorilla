from typing import Tuple



def move_vector_by_direction(vector: Tuple[int, int], direction: str, distance: int) -> Tuple[int, int]:

    """Calculates a new vector based on the given vector, direction, and distance.

    Args:

        vector: A tuple of two integers representing the x and y coordinates of the vector.

        direction: A string representing the direction of movement (north, south, east, or west).

        distance: A positive integer representing the distance to move in the given direction.

    """

    new_vector = list(vector)

    if direction == 'north':

        new_vector[1] += distance

    elif direction == 'south':

        new_vector[1] -= distance

    elif direction == 'east':

        new_vector[0] += distance

    elif direction == 'west':

        new_vector[0] -= distance

    else:

        raise ValueError("Invalid direction")

    return tuple(new_vector)

