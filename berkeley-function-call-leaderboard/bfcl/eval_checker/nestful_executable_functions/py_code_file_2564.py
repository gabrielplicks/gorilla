from typing import List, Tuple



def split_and_create_dict(data: List[str]) -> Dict[str, Tuple[str, str]]:

    """Splits a data set by commas and creates a dictionary from it.



    Args:

        data: A list of strings representing the data set. Each string is a row of the data set,

              and the elements of the row are separated by commas.



    Returns:

        A dictionary where the keys are the first element of each row in the data set, and the

        values are tuples of the other elements of the row.

    """

    data_split = [x.split(',') for x in data]

    dict_data = {x[0]: (x[1], x[2]) for x in data_split}



    return dict_data

