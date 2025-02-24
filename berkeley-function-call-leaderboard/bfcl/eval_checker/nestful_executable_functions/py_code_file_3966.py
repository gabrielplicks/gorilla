from collections import defaultdict

from typing import Dict, List, Any



def count_by_city(data: List[Dict[str, Any]]) -> Dict[str, int]:

    """Counts the number of records by city.



    Args:

        data: A list of objects with a 'city' attribute.



    Returns:

        A dictionary where the keys are city names and the values are the counts.

    """

    counts = defaultdict(int)

    for item in data:

        counts[item['city']] += 1

    return dict(counts)

