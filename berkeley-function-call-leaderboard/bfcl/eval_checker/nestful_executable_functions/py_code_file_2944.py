import itertools

from typing import List, Tuple



def create_attribute_list(X: itertools) -> List[Tuple[str, int]]:

    """Creates a list of tuples containing the attributes of a dataset.



    Args:

        X: The dataset.



    Returns:

        A list of tuples containing the attributes of the dataset.

    """

    columns = X.columns

    rows = X.index

    attribute_list = []

    for col in columns:

        for row in rows:

            attribute_list.append((col, row))



    return attribute_list

