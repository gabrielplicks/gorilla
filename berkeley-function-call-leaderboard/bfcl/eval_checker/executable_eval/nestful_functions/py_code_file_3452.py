from typing import Tuple



def calculate_bpm(song_length_sec: float, num_beats: int) -> float:

    """Calculates the BPM (beats per minute) of a song given its length in seconds and the number of beats it has.

    Args:

        song_length_sec: The length of the song in seconds.

        num_beats: The number of beats in the song.

    """

    minutes_in_a_second = 1/60

    song_length_minutes = song_length_sec * minutes_in_a_second

    bpm = num_beats / song_length_minutes

    return bpm

