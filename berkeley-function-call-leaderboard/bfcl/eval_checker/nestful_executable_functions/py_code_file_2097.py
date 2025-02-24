from typing import List, Dict



class Record:

    def __init__(self, id: int, status: str):

        self.id = id

        self.status = status



def filter_records(records: List[Record], params: Dict[str, str]) -> List[Record]:

    """Filters a list of records based on specified parameters.



    Args:

        records: A list of records, each with `id` and `status` attributes.

        params: A dictionary of filtering parameters.



    Returns:

        A list of records that match all filtering parameters.

    """

    def is_match(record: Record) -> bool:

        for key, value in params.items():

            if not hasattr(record, key) or getattr(record, key) != value:

                return False

        return True



    matches = [record for record in records if is_match(record)]

    return matches

