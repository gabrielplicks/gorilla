import sqlite3

from typing import List, Tuple



def get_articles(db_file: str, query: str) -> List[Tuple[int, str, str]]:

    """

    Returns a list of all articles that match the query.

    Each article is represented as a tuple (id, title, content) and sorted by the article ID in ascending order.



    Args:

        db_file: The SQLite database file name.

        query: The query string.

    """

    articles = []

    with sqlite3.connect(db_file) as conn:

        cursor = conn.cursor()

        cursor.execute(query)

        articles = cursor.fetchall()



    return sorted(articles, key=lambda article: article[0])

