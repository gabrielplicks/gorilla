from typing import List, Tuple



def has_user_read_article(user_id: int, article_id: int, user_article_pairs: List[Tuple[int, int]]) -> bool:

    """Checks whether the user has already read the article.



    Args:

        user_id: The ID of the user.

        article_id: The ID of the article.

        user_article_pairs: A list of tuples representing user-article pairs.

    """

    for pair in user_article_pairs:

        if pair[0] == user_id and pair[1] == article_id:

            return True

    return False

