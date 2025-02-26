from typing import List, Dict, Tuple, Union



def convert_table(table: List[Dict[str, Union[int, float, str, bool]]], conversions: List[Tuple[str, type]]) -> List[Dict[str, Union[int, float, str, bool]]]:

    """Converts some columns of a table to a different type.



    Args:

        table: A list of dictionaries representing the table.

        conversions: A list of tuples, where each tuple contains the name of a column to be converted and the target data type.



    Returns:

        The table with the specified columns converted to the target data types.

    """

    for row in table:

        for col, target_type in conversions:

            if col in row and type(row[col]) != target_type:

                row[col] = target_type(row[col])



    return table

