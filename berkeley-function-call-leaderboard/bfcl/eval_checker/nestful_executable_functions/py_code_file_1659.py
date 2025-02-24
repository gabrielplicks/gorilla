from typing import Dict



def create_movie_metadata_object(title: str, description: str, genre: str, year: int) -> Dict[str, str]:

    """Creates a movie metadata object with the given title, description, genre, and year.



    Args:

        title: The title of the movie.

        description: The description of the movie.

        genre: The genre of the movie.

        year: The year the movie was released.



    Returns:

        A dictionary containing the movie metadata.

    """

    metadata = {

        "Title": title,

        "Description": description,

        "Genre": genre,

        "Year": str(year)

    }

    return metadata

