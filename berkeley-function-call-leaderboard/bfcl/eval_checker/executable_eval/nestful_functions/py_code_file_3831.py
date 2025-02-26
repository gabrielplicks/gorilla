from typing import List, Dict



def filter_players(records: List[Dict], criteria: Dict) -> List[Dict]:

    """Filters a list of dictionaries based on some criteria.



    Args:

        records: A list of dictionaries representing player records from a database.

        criteria: A dictionary of search criteria. The keys correspond to database column names and the values are lists of acceptable values.



    Returns:

        A list of dictionaries that meet all the criteria.

    """

    filtered_records = []

    for record in records:

        is_valid = True

        for column, values in criteria.items():

            if record.get(column) not in values:

                is_valid = False

                break

        if is_valid:

            filtered_records.append(record)

    return filtered_records

