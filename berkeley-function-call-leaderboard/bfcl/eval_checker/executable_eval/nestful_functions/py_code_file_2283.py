import json

from typing import List



def read_and_format_data() -> str:

    """Reads the contents of the file 'data.txt' and returns a formatted JSON string.



    If the file is not found, the function returns an empty string.



    Args:

        None



    Returns:

        A formatted JSON string.

    """

    try:

        with open('data.txt', 'r') as f:

            data: List[str] = f.readlines()

            formatted_data = [{'name': name.strip(), 'age': int(age.strip())} for name, age in (line.split(',') for line in data)]

            return json.dumps(formatted_data)

    except FileNotFoundError:

        return ""

