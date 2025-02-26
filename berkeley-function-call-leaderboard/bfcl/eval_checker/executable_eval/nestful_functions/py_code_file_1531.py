from typing import List, Tuple, Union



def convert_to_discrete(continuous_value: float, index_bounds: List[Tuple[int, int]]) -> Union[int, None]:

    """Converts a continuous value into a discrete index based on a list of index bounds.



    Args:

        continuous_value: The continuous value to convert.

        index_bounds: A list of tuples representing the lower and upper bounds of each discrete range.



    Returns:

        The index of the discrete range that the continuous value falls into, or `None` if the value falls outside of all bounds.

    """

    for index, bounds in enumerate(index_bounds):

        lower, upper = bounds

        if continuous_value >= lower and continuous_value <= upper:

            return index

    return None

