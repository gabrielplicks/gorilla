from typing import Union



def midi_to_piano(midi_note: Union[int, float]) -> int:

    """Converts a MIDI note number to a piano key number.

    Args:

        midi_note: The MIDI note number.

    """

    return int(midi_note) - 21

