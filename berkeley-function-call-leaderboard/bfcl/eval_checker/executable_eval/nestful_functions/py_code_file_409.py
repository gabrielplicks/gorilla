from typing import List, Dict



def prepare_dataset(query_result: List[List[str]]) -> List[Dict[str, str]]:

    """Prepares a dataset from a query result.



    Args:

        query_result: A nested list where each element is a list of strings corresponding to the desired columns in the dataset.



    Returns:

        A list of dictionaries, where each dictionary contains the necessary information for the specific dataset you are preparing.

    """

    dataset = []



    for row in query_result:

        data = {}

        data['id'] = row[0]

        data['name'] = row[1]

        data['city'] = row[2]

        dataset.append(data)



    return dataset

