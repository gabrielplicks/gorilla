from typing import List



def count_distinct_colors(list_of_colors: List[str]) -> int:

    """Counts the number of distinct colors in a list of colors.

    Args:

        list_of_colors: A list of colors represented by strings.

    """

    distinct_colors = set()

    for color in list_of_colors:

        distinct_colors.add(color)

    return len(distinct_colors)

