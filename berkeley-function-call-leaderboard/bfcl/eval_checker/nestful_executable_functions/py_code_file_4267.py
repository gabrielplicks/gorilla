from typing import List, Dict



def sort_by_confidence(prediction_associations: List[Dict[str, float]]) -> List[Dict[str, float]]:

    """Sorts the records by the confidence of their predictions, but only considers records where the prediction confidence is greater than or equal to 0.5.



    The final sorted list only contains the prediction associations where the confidence is at least 0.5 and the prediction is not `None`.



    Args:

        prediction_associations: A list of prediction association records.



    Returns:

        A new list of records sorted by the confidence of their predictions.

    """

    filtered_records = filter(lambda record: record['confidence'] >= 0.5, prediction_associations)

    sorted_records = sorted(filtered_records, key=lambda record: record['confidence'], reverse=True)

    return sorted_records

