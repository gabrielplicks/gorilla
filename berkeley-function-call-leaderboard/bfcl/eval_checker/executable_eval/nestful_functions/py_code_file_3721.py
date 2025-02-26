from typing import List, Dict



def generate_metrics(input_list: List[Dict[str, int]]) -> List[Dict[str, str]]:

    """Generates a list of metrics with timestamp, value, and unit attributes.



    Args:

        input_list: A list of objects (or dictionaries) with timestamp and value attributes.



    Returns:

        A list of objects (or dictionaries) with timestamp, value, and unit attributes.

    """

    return [{'timestamp': obj['timestamp'], 'value': obj['value'], 'unit': 'seconds' if obj['value'] < 1024 else 'bytes'} for obj in input_list]

