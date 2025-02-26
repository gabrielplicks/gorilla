from typing import List



def calculate_ber(tpr_list: List[float], fpr_list: List[float]) -> List[float]:

    """Calculates the balanced error rate (BER) of a classifier given its true positive rate (TPR) and false positive rate (FPR).



    Args:

        tpr_list: A list of true positive rates.

        fpr_list: A list of false positive rates.



    Returns:

        A list of balanced error rates.

    """

    fnr_list = [1 - tpr for tpr in tpr_list]  # Calculate FNR

    ber_list = [(1 - fpr) + fnr / 2 for fpr, fnr in zip(fpr_list, fnr_list)]  # Calculate BER

    return ber_list

