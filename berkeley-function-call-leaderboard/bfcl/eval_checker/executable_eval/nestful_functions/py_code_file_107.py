from typing import List, Dict, Tuple, Union



def add_enum(data: Union[List[str], None]) -> Tuple[Union[List[str], None], Dict[int, str]]:

    """Adds an enumeration to a list of strings.



    Args:

        data: A list of strings.



    Returns:

        A tuple of the data and a dictionary of index-to-string mappings.

    """

    if not data or data is None:

        return (None, {})



    index_mapping = dict()

    for index, value in enumerate(data):

        index_mapping[index] = value



    return (data, index_mapping)

