import re

from typing import List



def extract_product_ids(product_urls: List[str]) -> List[str]:

    """Extracts product IDs from a list of Amazon product URLs.



    Args:

        product_urls: A list of Amazon product URLs.



    Returns:

        A list of product IDs.

    """

    product_ids = []

    for url in product_urls:

        match = re.search(r"dp/(.+)", url)

        if match:

            product_ids.append(match.group(1))

    return product_ids

