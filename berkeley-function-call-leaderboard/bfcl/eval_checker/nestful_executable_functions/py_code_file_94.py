from typing import List, Dict



def remove_dupes_by_key(dicts: List[Dict], key: str) -> List[Dict]:

    """Removes duplicates from a list of dictionaries by comparing a specified key.

    Preserves the first occurrence of the duplicate.

    Args:

        dicts: A list of dictionaries.

        key: The key to compare for duplicates.

    """

    seen = set()

    unique_dicts = []



    for d in dicts:

        if d[key] not in seen:

            seen.add(d[key])

            unique_dicts.append(d)



    return unique_dicts

