from typing import Dict



def row_to_dict(row: Dict[str, any]) -> Dict[str, any]:

    """Converts a database table row into a dictionary with column names as keys.



    Args:

        row: The database table row.



    Returns:

        A dictionary with column names as keys and corresponding values.

    """

    return {col: row[col] for col in row.keys()}

