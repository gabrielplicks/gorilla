import sqlite3

from typing import List, Tuple



def construct_pagination_query(table_name: str, columns: List[str], page_number: int, page_size: int) -> Tuple[str, Tuple[int, int]]:

    """Constructs a SQL query for pagination with a specific page number and page size.



    Args:

        table_name: The name of the table to query.

        columns: The columns to query.

        page_number: The page number.

        page_size: The number of items per page.



    Returns:

        A tuple containing the SQL query string and the parameters for the query.

    """

    query = "SELECT %s FROM %s LIMIT ? OFFSET ?" % (",".join(columns), table_name)

    return query, (page_size, (page_number - 1) * page_size)

