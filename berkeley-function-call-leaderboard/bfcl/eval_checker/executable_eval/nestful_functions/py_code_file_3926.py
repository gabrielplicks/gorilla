from typing import List, Tuple



def compute_relative_coordinates(coordinates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:

    """Computes the relative coordinates of the elements in one list with respect to a fixed reference point.

    Args:

        coordinates: A list of lists, each representing a coordinate.

    Raises:

        Exception: If the input list is empty or elements do not have two coordinates.

    """

    if not coordinates or len(coordinates[0]) != 2:

        raise Exception("Invalid input: List is empty or elements do not have two coordinates.")



    reference_point = coordinates[0]

    relative_coordinates = []



    for element in coordinates:

        if len(element) != 2:

            raise Exception("Invalid input: One of the elements does not have two coordinates.")

        relative_coordinates.append((element[0] - reference_point[0], element[1] - reference_point[1]))



    return relative_coordinates

