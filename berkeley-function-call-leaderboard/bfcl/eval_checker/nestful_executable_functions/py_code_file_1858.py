from typing import Dict, List



def create_label_dictionary(labels: List[str]) -> Dict[int, str]:

    """Creates a dictionary of labels from a list of strings.

    The dictionary is constructed by extracting the labels from the `labels` list based on their indices.

    Args:

        labels: A list of strings.

    Returns:

        A dictionary where each key is an integer index and each value is a string label.

    """

    label_dict = {}

    for i, label in enumerate(labels):

        label_dict[i] = label

    return label_dict

