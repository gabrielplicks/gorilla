from typing import List, Dict



def extract_results(basc_field_result: Dict) -> List[Dict]:

    """Extracts the results from the `basc_field_result` object.



    Args:

        basc_field_result: A dictionary containing the results.



    Returns:

        A list of dictionaries, where each dictionary represents a single result,

        and each dictionary contains two keys: `name` (the name of the result)

        and `value` (the value of the result).

    """

    results = basc_field_result['results']

    output_results = []

    for result in results:

        output_results.append({'name': result['name'], 'value': result['value']})

    return output_results

