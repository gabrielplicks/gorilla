from typing import List



def generate_labels(values: List[str]) -> List[str]:

    """

    Generates a list of labels by appending "label_" and the index number to each

    string in `values`. For example, given `values = ['a', 'b', 'c']`, this function

    returns `['label_0_a', 'label_1_b', 'label_2_c']`.



    Args:

        values: A list of strings.



    Raises:

        ValueError: If `values` is not a list of strings.

    """

    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):

        raise ValueError("Input must be a list of strings.")



    labels = []

    for i, value in enumerate(values):

        labels.append(f"label_{i}_{value}")

    return labels

