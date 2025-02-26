from typing import Dict, List



def transform_logs(logs: List[Dict[str, int]]) -> Dict[str, List[int]]:

    """Transforms a list of logs into a dictionary where the keys are the keys in the dictionaries and the values are lists with the values in the corresponding dictionaries.



    Args:

        logs: A list of dictionaries where each dictionary represents a log.



    Returns:

        A dictionary where the keys are the keys in the dictionaries and the values are lists with the values in the corresponding dictionaries.

    """

    transformed_logs = {key: [] for key in logs[0].keys()}



    for log in logs:

        for key, value in log.items():

            transformed_logs[key].append(value)



    return transformed_logs

