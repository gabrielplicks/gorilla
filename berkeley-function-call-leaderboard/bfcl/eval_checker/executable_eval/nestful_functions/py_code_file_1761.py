from typing import Dict, List, Any



def transpose_df(rows: List[Dict[str, Any]]) -> Dict[str, List[Any]]:

    """Transposes a list of dictionaries representing rows in a data frame into a dictionary.



    The keys of the resulting dictionary are the column names and the values are lists of all the values for that column.



    Args:

        rows: A list of dictionaries where each dictionary represents a row in a data frame.



    Returns:

        A dictionary where the keys are the column names and the values are lists of all the values for that column.

    """

    return {k: [d[k] for d in rows] for k in rows[0].keys()}

