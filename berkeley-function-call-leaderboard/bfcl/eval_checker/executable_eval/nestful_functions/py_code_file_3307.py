import wave

from typing import Tuple



def extract_audio_params(audio_file_path: str) -> Tuple[int, int, int]:

    """Extracts audio parameters from a given audio file.



    Args:

        audio_file_path: The path to the audio file.



    Returns:

        A tuple containing the number of channels, sample rate, and bit depth.

    """

    with wave.open(audio_file_path, 'r') as audio_file:

        channels = audio_file.getnchannels()

        sample_rate = audio_file.getframerate()

        bit_depth = audio_file.getsampwidth() * 8

    return (channels, sample_rate, bit_depth)

