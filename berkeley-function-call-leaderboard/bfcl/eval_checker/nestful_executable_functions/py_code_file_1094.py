from typing import List, Dict



def create_query_result(rows: List[Dict[str, str]], columns: List[str]) -> List[Dict[str, str]]:

    """Creates a list of objects representing the result of a database query.



    Each object is a dictionary with keys corresponding to the column names and values

    corresponding to the column values.



    Args:

        rows: A list of query results, where each result is a dictionary representing a row.

        columns: A list of strings representing the column names.



    Returns:

        A list of objects where each object is a dictionary with keys corresponding to the

        column names and values corresponding to the column values.

    """

    result = []



    for row in rows:

        obj = {}

        for column in columns:

            obj[column] = row[column]

        result.append(obj)



    return result

