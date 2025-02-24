import re



def get_primary_key(table_name: str) -> str:

    """Extracts the primary key from a table name.



    Args:

        table_name: The name of the table.



    Returns:

        The primary key of the table, or None if the table name does not match the expected pattern.

    """

    match = re.match(r'^(\w+)_\w+$', table_name)

    if match:

        return match.group(1)

    else:

        return None

