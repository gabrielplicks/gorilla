import json

from typing import List, Dict



def get_database_schemas() -> List[Dict]:

    """Retrieves a list of database schemas from a JSON file.



    Each schema is a dictionary.



    Args:

        None



    Returns:

        A list of schemas, where each schema is a dictionary.



    Raises:

        FileNotFoundError: If the JSON file is not found.

        json.decoder.JSONDecodeError: If the data in the JSON file is malformed.

    """

    try:

        with open('database_schemas.json', 'r') as file:

            data = json.load(file)

            schemas = data['schemas']

            return schemas

    except FileNotFoundError:

        print('Error: Database schemas file not found.')

    except json.decoder.JSONDecodeError:

        print('Error: Database schemas file is malformed.')

