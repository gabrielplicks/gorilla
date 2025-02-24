from datetime import datetime

from typing import List, Dict



def get_urls_with_expiry_time(urls: List[Dict[str, str]]) -> List[Dict[str, str]]:

    """

    Returns a list of URLs with expiry time greater than the current time.

    Args:

        urls: A list of dictionary objects representing URLs with their metadata.

    """

    urls_with_expiry_time = []

    now = datetime.now()

    for url in urls:

        parsed_expiry_time = datetime.strptime(url['expiry_time'], "%Y-%m-%dT%H:%M:%S")

        if parsed_expiry_time > now:

            urls_with_expiry_time.append(url)

    return urls_with_expiry_time

