from typing import Dict, List, Tuple

import json



def count_age_entries(json_file: str) -> List[Tuple[int, int]]:

    """Reads a JSON file containing a list of entry objects with "name" and "age" fields.

    Returns a list of tuples containing the number of occurrences of each "age" value in the file, sorted in increasing order by age.

    Args:

        json_file: The path to the JSON file.

    """

    with open(json_file) as f:

        entries = json.loads(f.read())



    age_counts: Dict[int, int] = {}

    for entry in entries:

        age = entry['age']

        age_counts[age] = age_counts.get(age, 0) + 1



    return sorted(age_counts.items())

