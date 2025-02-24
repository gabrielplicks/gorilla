from typing import List, Tuple



def calculate_sensitivity(confusion_matrix: List[List[int]]) -> float:

    """Calculates the sensitivity (or TPR) of the model given the confusion matrix.



    Args:

        confusion_matrix: A 2D array representing the confusion matrix with the form

            [

                [TN, FP],

                [FN, TP]

            ]

    """

    TN, FP, FN, TP = confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0], confusion_matrix[1][1]

    sensitivity = TP / (TP + FN)

    return sensitivity

