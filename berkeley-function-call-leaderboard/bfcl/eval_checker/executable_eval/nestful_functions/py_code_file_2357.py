from typing import List, Dict



def filter_comments(comments: List[Dict[str, int]], author_ids: List[int]) -> List[Dict[str, int]]:

    """Filters a list of comments to include only those with authors from a given list of author IDs.



    Args:

        comments: A list of dictionaries, where each dictionary contains a comment ID and an author ID.

        author_ids: A list of author IDs.



    Returns:

        A list of dictionaries, where each dictionary contains a comment ID and an author ID, and the author ID is in the given list of author IDs.

    """

    def check_author(comment: Dict[str, int]) -> bool:

        return comment['author_id'] in author_ids

    return list(filter(check_author, comments))

