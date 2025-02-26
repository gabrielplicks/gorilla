from typing import List, Dict



def get_records_by_role_and_department(data: List[Dict], role_id: int) -> List[Dict]:

    """

    Returns a list of dictionaries containing only the records that have the given role_id and whose department_name is "Sales".

    Args:

        data: A list of dictionaries representing records in a database.

        role_id: The role_id to filter the records by.

    """

    filtered_records = [record for record in data

                        if record['role_id'] == role_id and record['department_name'] == 'Sales']



    output = [{

        'user_id': record['user_id'],

        'user_name': record['user_name']

    } for record in filtered_records]



    return output

