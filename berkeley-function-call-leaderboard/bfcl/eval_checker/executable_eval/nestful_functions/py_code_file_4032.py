from typing import List



def adjust_predictions(predictions: List[float]) -> List[int]:

    """Adjusts predictions based on a threshold.



    Args:

        predictions: A list of float predictions.



    Returns:

        A list of int predictions adjusted based on the threshold.

    """

    adjusted_predictions = []



    for prediction in predictions:

        if prediction < 0.5:

            adjusted_predictions.append(0)

        else:

            adjusted_predictions.append(1)



    return adjusted_predictions

