from typing import List



def count_labels(labels: List[int]) -> tuple:

    """Counts the number of overall labels, the number of positive labels, and the number of negative labels in the given set.



    Args:

        labels: A list of labels. The labels are either 0 or 1.



    Returns:

        A tuple containing the total number of labels, the number of positive labels, and the number of negative labels.

    """

    counter = {"total_labels": 0, "positive_labels": 0, "negative_labels": 0}

    for label in labels:

        counter["total_labels"] += 1

        if label == 1:

            counter["positive_labels"] += 1

        else:

            counter["negative_labels"] += 1

    return tuple(counter.values())

