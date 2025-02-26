from typing import List, Tuple, Dict



def create_dictionary_from_database_query(results: List[Tuple]) -> Dict[str, List]:

    """Creates a dictionary of key-value pairs from the results of a database query.

    The keys represent the column names of the query result and the values represent a list of corresponding values in the rows.

    Args:

        results: A list of tuples representing the query results. Each tuple represents a row and contains the corresponding values for all columns.

    """

    column_names = ["col1", "col2", "col3"]  # Example column names

    dictionary = {key: [] for key in column_names}

    for row in results:

        values = [row[i] for i in range(len(row))]

        for i in range(len(column_names)):

            column_name = column_names[i]

            dictionary[column_name].append(values[i])

    return dictionary

