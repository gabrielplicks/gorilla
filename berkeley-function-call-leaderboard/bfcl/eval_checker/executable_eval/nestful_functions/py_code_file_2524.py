from typing import Dict, List, Tuple



def parse_movies_by_category(path: str) -> Dict[str, List[Tuple[str, str]]]:

    """Parses a file containing movie information and categorizes movies by their genres.



    Args:

        path: The file path containing movie information.



    Returns:

        A dictionary with each category as a key and a list of movies as the value.

        Each movie is represented as a tuple containing the movie title and release date.

    """

    categories = {}

    with open(path, 'r') as file:

        for line in file:

            movie_title, category, release_date = line.strip().split(',')

            if category not in categories:

                categories[category] = []

            categories[category].append((movie_title, release_date))

    return categories

