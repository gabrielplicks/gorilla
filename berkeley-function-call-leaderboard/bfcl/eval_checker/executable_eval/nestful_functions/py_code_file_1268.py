from typing import List, Tuple



def partition_data(data: List[Tuple[int, int, float]], threshold: float) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:

    """Partitions a list of 3-tuples into two sublists based on a threshold value for responsibilities.



    Args:

        data: A list of 3-tuples, where each tuple represents a data point with two numeric values.

            The first value is an ID number, the second value is a target value, and the third value is the responsibility of a machine learning model for predicting the target value.

        threshold: A threshold value for responsibilities.



    Returns:

        A tuple containing two sublists: one with the tuples for which the model has a responsibility below the threshold and one with the tuples for which the model has a responsibility above or equal to the threshold.

    """

    return [x for x in data if x[2] < threshold], [x for x in data if x[2] >= threshold]

