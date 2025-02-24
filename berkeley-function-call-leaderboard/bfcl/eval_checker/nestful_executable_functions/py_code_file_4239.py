from typing import List, Dict, Any



def remap_keys(columns: List[str], data: Dict[str, List[Any]]) -> Dict[str, List[Any]]:

    """Remaps the keys in a dictionary to match the column names in a given list.



    Args:

        columns: A list of column names.

        data: A dictionary where the keys are column names and the values are the corresponding data.



    Returns:

        A dictionary with the keys remapped to the column names in `columns`.

    """

    remapped_data = {}

    for column in columns:

        remapped_data[column] = data.get(column, [])

    return remapped_data

