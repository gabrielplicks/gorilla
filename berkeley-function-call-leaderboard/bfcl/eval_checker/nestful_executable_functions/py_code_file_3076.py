from typing import List, Tuple



def substring_count_at_points(input_string: str, points: List[Tuple[int, int]]) -> dict:

    """Calculates the number of occurrences of a substring at each point in a list of points.



    Args:

        input_string: The input string.

        points: A list of points, where each point is a tuple of two integers representing the coordinates of the point.



    Returns:

        A dictionary where each key is the string representation of a point and the corresponding value is the number of occurrences of the substring in the input string.

    """

    result = {}

    for point in points:

        x, y = point

        substring = input_string[x:x+y]

        result[str(point)] = input_string.count(substring)

    return result

