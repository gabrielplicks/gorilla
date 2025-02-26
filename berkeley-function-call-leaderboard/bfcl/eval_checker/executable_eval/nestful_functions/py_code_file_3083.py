from typing import List, Dict, Tuple



def process_api_responses(original_list: List[str]) -> List[Tuple[str, int]]:

    """Processes a list of API responses and returns a list of key-value pairs, where the keys are the specific API responses and the values are the number of occurrences in the original list.



    Args:

        original_list: A list of API responses.



    Returns:

        A list of key-value pairs, where the keys are the API responses and the values are the number of occurrences in the original list.

    """

    response_counts: Dict[str, int] = {}

    for response in original_list:

        response_counts[response] = response_counts.get(response, 0) + 1

    return list(response_counts.items())

