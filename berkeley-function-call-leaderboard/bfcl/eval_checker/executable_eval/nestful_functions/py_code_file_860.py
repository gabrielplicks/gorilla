from typing import List



def calculate_false_positive_rate(probabilities: List[float]) -> float:

    """Calculates the false positive rate for a given list of probabilities.



    Args:

        probabilities: A list of probabilities.



    Returns:

        The calculated false positive rate.

    """

    false_positives = 0

    threshold = 0.5



    for probability in probabilities:

        if probability > threshold:

            false_positives += 1



    false_positive_rate = false_positives / len(probabilities)



    return false_positive_rate

