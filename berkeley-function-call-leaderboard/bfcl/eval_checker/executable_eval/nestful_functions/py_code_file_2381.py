from typing import Dict



page_rank_scores: Dict[str, float] = {

    "https://www.google.com": 0.5,

    "https://www.wikipedia.org": 0.25,

    "https://www.amazon.com": 0.15,

}



def get_page_rank_score(url: str) -> float:

    """Calculates the PageRank score of a given URL.

    Args:

        url: The URL of the page.

    Returns:

        The PageRank score of the page. If the page is not found or if the page is not part of the database,

        the function returns 0.0.

    """

    if url in page_rank_scores:

        return page_rank_scores[url]

    else:

        return 0.0

