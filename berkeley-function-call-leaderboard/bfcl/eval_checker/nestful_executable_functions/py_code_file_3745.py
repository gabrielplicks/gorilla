from typing import Dict



def generate_filter_string(filter_args: Dict[str, str]) -> str:

    """Generates a filter string for a database query based on the provided filter arguments.



    Args:

        filter_args: A dictionary containing the filter arguments.



    Returns:

        A filter string that can be used in a SQL WHERE clause.

    """

    filter_string = []

    for key, value in filter_args.items():

        if isinstance(value, str):

            filter_string.append(f"{key} = '{value}'")

        elif isinstance(value, bool):

            filter_string.append(f"{key} = {str(value).lower()}")

        else:

            filter_string.append(f"{key} = {value}")

    return " AND ".join(filter_string)

