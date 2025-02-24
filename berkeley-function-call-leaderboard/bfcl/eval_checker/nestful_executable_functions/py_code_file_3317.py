from collections import defaultdict

from typing import List, Dict



class Record:

    def __init__(self, k: str, v: int):

        self.k = k

        self.v = v



def group_records_by_k(records: List[Record]) -> Dict[str, List[Record]]:

    """Groups a list of Record objects by the 'k' property and sorts each group by the 'v' property.



    Args:

        records: A list of Record objects.



    Returns:

        A dictionary where the keys are the 'k' properties and the values are lists of Record objects sorted by the 'v' property.

    """

    grouped_records = defaultdict(list)

    for record in records:

        grouped_records[record.k].append(record)



    for records_list in grouped_records.values():

        records_list.sort(key=lambda r: r.v)



    return grouped_records

