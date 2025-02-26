from typing import List, Dict



def count_apples(object_detection_results: List[Dict], threshold: float) -> int:

    """Counts the number of apples detected with a confidence score higher than a given threshold.



    Args:

        object_detection_results: A list of dictionaries containing object detection results.

        threshold: The minimum confidence score for an object to be considered an apple.

    """

    num_apples = 0



    for detection in object_detection_results:

        object_name = detection['object_name']

        confidence_score = detection['confidence_score']

        bbox_coordinates = detection['bbox_coordinates']



        if object_name == 'apple' and confidence_score > threshold:

            num_apples += 1



    return num_apples

