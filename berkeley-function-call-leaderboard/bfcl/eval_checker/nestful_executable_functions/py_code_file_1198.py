import json



def load_categories(file_path: str) -> dict:

    """

    Loads a JSON file and creates a dictionary from the contents.

    The JSON file should have a line-delimited format, where each line is a JSON object with a `name` and `id` field.

    The function returns a dictionary where the `name` is the key and the `id` is the value.



    Args:

        file_path: The path to the JSON file.

    """

    categories = {}

    with open(file_path, 'r') as file:

        for line in file:

            category = json.loads(line)

            categories[category['name']] = category['id']



    return categories

