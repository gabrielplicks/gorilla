from typing import List, Dict

import datetime



def create_document_keys(documents: List[Dict[str, str]]) -> List[str]:

    """Creates a list of keys for a given list of documents.



    Each key is formatted as `{date_created}_{document_id}` where the date is in the format of `YYYYMMDD`.



    Args:

        documents: A list of documents where each document is a dictionary with keys 'date_created' and 'document_id'.



    Returns:

        A list of keys.

    """

    keys = []

    for document in documents:

        date_created = datetime.datetime.strptime(document['date_created'], '%Y%m%d').date()

        key = f'{date_created}_{document["document_id"]}'

        keys.append(key)

    return keys

