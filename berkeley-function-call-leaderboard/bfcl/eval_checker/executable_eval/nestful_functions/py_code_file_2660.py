from typing import List, Dict



def get_data_subset(data: List[Dict], attributes: List[str]) -> List[List]:

    """

    Returns a list of lists containing the corresponding values for each attribute in the data.



    Args:

        data: A list of dictionaries containing the data.

        attributes: A list of strings representing the attributes to extract.

    """

    return [[d[attr] for attr in attributes] for d in data]

