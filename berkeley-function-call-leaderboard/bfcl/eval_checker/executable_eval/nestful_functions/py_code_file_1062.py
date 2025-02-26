import re



def clean_column_names(column_name: str) -> str:

    """Cleans column names by removing non-alphanumeric characters and converting them to lowercase.



    Args:

        column_name: The input column name.



    Returns:

        A string with only alphanumeric characters, underscores, and lowercase letters.

    """

    column_name = re.sub(r'[^a-zA-Z0-9_]', '', column_name).lower()

    return column_name

