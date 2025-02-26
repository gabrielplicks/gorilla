import re

from typing import Union



table_name_pattern = re.compile(r'^/.*\/.*\/(.*)$')



def extract_table_name(table_name: Union[str, bytes]) -> str:

    """Extracts and returns the table name from a given table name string.



    The table name string must have a valid format of the following form:

    /database_name/schema_name/table_name



    Args:

        table_name: The table name string.



    Raises:

        ValueError: If the table name format is invalid.

    """

    match = table_name_pattern.match(table_name)

    if match:

        return match.group(1)

    else:

        raise ValueError('Invalid table name format')

