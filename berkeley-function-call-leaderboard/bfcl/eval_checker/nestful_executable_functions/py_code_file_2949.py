from typing import List, Dict



def get_knobs_values(data: List[Dict[str, List[int]]]) -> List[Dict[str, List[int]]]:

    """

    Returns a new list of dictionaries containing only the 'knobs' key and its value from each dictionary.

    The output list preserves the order of the input list.



    Args:

        data: A list of dictionaries.



    Returns:

        A new list of dictionaries containing only the 'knobs' key and its value from each dictionary.

    """

    return [{'knobs': d['knobs']} for d in data if 'knobs' in d]

