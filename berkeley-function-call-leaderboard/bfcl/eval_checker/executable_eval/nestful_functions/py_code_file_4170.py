from typing import List



def predict_score(features: List[float], coefficients: List[float]) -> float:

    """Calculates the weighted sum of a list of features using a list of coefficients.



    Args:

        features: A list of features.

        coefficients: A list of coefficients corresponding to the features.

    """

    score = 0

    for feature, coefficient in zip(features, coefficients):

        score += feature * coefficient

    return score

