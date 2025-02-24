from typing import List, Dict



def generate_sql_insert_query(records: List[Dict[str, str]]) -> str:

    """Constructs a SQL query that does an `INSERT` into a table named `my_table` with the given records.

    If the column names are not provided explicitly, they should be derived from the first dictionary in the list.

    Args:

        records: A list of dictionaries representing database records.

    """

    columns = list(records[0].keys())

    query = f"INSERT INTO my_table ({', '.join(columns)}) VALUES"

    for record in records:

        values = [str(record[col]) for col in columns]

        query += f" ({', '.join(values)}),"

    return query[:-1]

