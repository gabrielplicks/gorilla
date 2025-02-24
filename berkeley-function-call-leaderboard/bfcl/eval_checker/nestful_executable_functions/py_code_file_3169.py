from typing import List



def transform_to_tsv(csv_string: str) -> str:

    """Transforms a string of comma-separated values into a string of tab-separated values.

    Args:

        csv_string: A string of comma-separated values.

    """

    rows: List[str] = csv_string.splitlines()

    columns: List[List[str]] = [row.split(',') for row in rows]

    transformed_rows: List[str] = ['\t'.join(row) for row in columns]

    transformed_string: str = '\n'.join(transformed_rows)

    return transformed_string

