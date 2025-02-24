from typing import List, Dict



def filter_by_tag(data: List[Dict], tag: str = "tag1") -> List[Dict]:

    """

    Filters a list of dictionaries by a specific tag.



    Args:

        data: A list of dictionaries to filter.

        tag: The tag to filter by. Default is "tag1".



    Returns:

        A list of dictionaries that contain the specified tag.

    """

    filtered_results = []



    for d in data:

        if tag in d:

            filtered_results.append(d)



    return filtered_results

