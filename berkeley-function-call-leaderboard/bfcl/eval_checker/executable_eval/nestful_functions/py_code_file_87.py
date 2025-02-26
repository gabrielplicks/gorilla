from typing import List, Tuple



def filter_rows_by_group(rows: List[Tuple], group_name: str) -> List[Tuple]:

    """

    Filters rows from a nested list of rows based on the values in a specified group.

    Returns a list of tuples containing the values from the specified group and the filtered rows.

    Args:

        rows: A nested list of rows.

        group_name: The name of the group to filter by.

    """

    filtered_rows = list(filter(lambda row: row[0] == group_name, rows))

    return filtered_rows

