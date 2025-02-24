from typing import List, Dict, Any



def table_to_dict(table: List[List[Any]]) -> List[Dict[str, Any]]:

    """Converts a two-dimensional list table into a dictionary.



    The first row of the table is assumed to be the header row, where the keys of the dictionary are extracted.

    The remaining rows are iterated over to construct a dictionary for each data row.

    The values corresponding to the keys are assigned based on the respective indices in the header row.



    Args:

        table: A two-dimensional list representing the table data.



    Returns:

        A list of dictionaries representing the table data.

    """

    header = table[0]

    data = table[1:]



    dict_table = []

    for row in data:

        row_dict = {}

        for i, item in enumerate(row):

            row_dict[header[i]] = item

        dict_table.append(row_dict)



    return dict_table

