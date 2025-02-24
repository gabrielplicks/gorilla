from typing import List, Dict, Tuple



def extract_labels_from_json(json_objects: List[Dict]) -> List[Tuple[str, str]]:

    """Extracts language and label pairs from a list of JSON objects.



    Args:

        json_objects: A list of JSON objects, each containing a "language" and "label" field.



    Returns:

        A list of tuples containing the language and label for each object.

    """

    labels = []

    for obj in json_objects:

        language = obj.get('language')

        label = obj.get('label')

        labels.append((language, label))

    return labels

