from typing import List



def format_labels(labels: List[str], label_length: int) -> List[str]:

    """Formats a list of labels by adding underscores to ensure all labels have a length of exactly `label_length`.



    Args:

        labels: A list of labels.

        label_length: The desired length of each label.



    Returns:

        A list of formatted labels, where each label is a string of exactly `label_length` characters.

    """

    formatted_labels = []

    for i, label in enumerate(labels):

        if len(label) < label_length:

            formatted_label = label + '_' * (label_length - len(label))

        else:

            formatted_label = label[:label_length]

        formatted_labels.append(formatted_label)

    return formatted_labels

