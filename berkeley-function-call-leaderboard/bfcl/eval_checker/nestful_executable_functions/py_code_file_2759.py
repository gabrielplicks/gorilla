from typing import List



def get_table_names_from_query(sql_query: str) -> List[str]:

    """Extracts the table names referenced in a SQL query string.



    Args:

        sql_query: The SQL query string.



    Returns:

        A list of table names referenced in the query.

    """

    tokens = sql_query.split()

    table_names = []

    join_keyword = False

    for i, token in enumerate(tokens):

        if token in ('JOIN', 'FROM'):

            join_keyword = True

            continue

        if join_keyword:

            if token == 'FROM':

                table_name = tokens[i + 1]

                table_names.append(table_name)

            else:

                table_name = token

                table_names.append(table_name)

            join_keyword = False



    return table_names

