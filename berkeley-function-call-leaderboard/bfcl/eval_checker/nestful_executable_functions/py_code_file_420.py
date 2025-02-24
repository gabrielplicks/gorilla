from typing import List, Dict



def process_csv_list(csv_list: List[str]) -> List[Dict[str, str]]:

    """Transforms a list of CSV-formatted strings into a list of dictionaries.



    Each CSV-formatted string contains the fields "name", "age", and "salary".



    Args:

        csv_list: A list of CSV-formatted strings.



    Returns:

        A list of dictionaries, where each dictionary contains the fields "name", "age", and "salary".

    """

    result = []

    for csv_string in csv_list:

        fields = csv_string.split(",")  # split by commas

        fields_dict = {

            "name": fields[0],

            "age": fields[1],

            "salary": fields[2]

        }

        result.append(fields_dict)

    return result

