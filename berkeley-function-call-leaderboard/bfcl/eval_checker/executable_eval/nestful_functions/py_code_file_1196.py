from typing import Dict



def parse_comments(comments: str) -> Dict[str, int]:

    """Parses a string of comments separated by newlines and returns a dictionary where each comment is a key and the value is the number of times the comment appears.



    Args:

        comments: A string of comments separated by newlines.



    Returns:

        A dictionary where the key is the comment and the value is the number of times the comment appears.

    """

    comment_list = comments.split('\n')

    comment_counts: Dict[str, int] = {}

    for comment in comment_list:

        key, value = comment.split(':')

        if value.isdigit():

            value = int(value)

        else:

            value = 1

        comment_counts[key] = value



    return comment_counts

