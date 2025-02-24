from typing import List



def remove_edge_artifacts(input_list: List[float]) -> float:

    """Removes edge artifacts from a time series signal.



    Args:

        input_list: A list of numbers representing the time series signal.



    Returns:

        The mean value of the time series signal after removing edge artifacts.

    """

    mean = sum(input_list) / len(input_list)

    while len(input_list) > 10:

        input_list = input_list[len(input_list) // 4:len(input_list) * 3 // 4]

    mean = sum(input_list) / len(input_list)

    return mean

