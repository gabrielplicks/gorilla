from collections import defaultdict

from typing import Dict, List



def partition_comments(comments: List[str]) -> Dict[str, List[str]]:

    """Partitions a list of comments into a dictionary mapping user_id to a list of comment_texts.



    Args:

        comments: A list of strings, each in the format `user_id|comment_id|comment_text`.



    Returns:

        A dictionary mapping `user_id` to a list of `comment_texts`.

    """

    comments_by_user = defaultdict(list)

    for comment in comments:

        user_id, comment_id, comment_text = comment.split('|')

        comments_by_user[user_id].append(comment_text)

    return comments_by_user

