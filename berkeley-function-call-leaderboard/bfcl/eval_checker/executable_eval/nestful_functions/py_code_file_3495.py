from typing import Dict



def merge_and_calculate_difference(actual: Dict[str, float], expected: Dict[str, float]) -> Dict[str, float]:

    """Merges two dictionaries `actual` and `expected`, and calculates the differences between the actual and expected values.



    Args:

        actual: A dictionary of actual time and date measurement values.

        expected: A dictionary of expected values.



    Returns:

        A dictionary with keys as timestamps and values as the differences between the actual and expected values.

    """

    difference = {}

    for timestamp, actual_value in actual.items():

        expected_value = expected[timestamp]

        difference[timestamp] = actual_value - expected_value

    return difference

