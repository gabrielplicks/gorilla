from typing import List, Union



def concatenate_points(points1: List[List[Union[int, float]]], points2: List[List[Union[int, float]]]) -> List[List[Union[int, float]]]:

    """

    Concatenate two lists of points, where each point is represented by a list of two numeric values.

    This function checks if each point is valid and ignores invalid points.



    Args:

        points1: The first list of points.

        points2: The second list of points.

    """

    concatenated_points = []

    for point in points1:

        if len(point) == 2 and all(isinstance(value, (int, float)) for value in point):

            concatenated_points.append(point)

    for point in points2:

        if len(point) == 2 and all(isinstance(value, (int, float)) for value in point):

            concatenated_points.append(point)

    return concatenated_points

