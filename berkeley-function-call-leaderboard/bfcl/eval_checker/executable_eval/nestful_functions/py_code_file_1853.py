from typing import List



def song_in_list(song: str, songs: List[str] = []) -> bool:

    """Checks if a song is in a list of songs.



    Args:

        song: The name of the song to check.

        songs: The list of songs to check against.



    Returns:

        True if the song is in the list, False otherwise.

        If the song name is not provided, the function returns True if the list is empty and False otherwise.

    """

    if song == '':

        return len(songs) == 0

    for song_name in songs:

        if song_name == song:

            return True

    return False

