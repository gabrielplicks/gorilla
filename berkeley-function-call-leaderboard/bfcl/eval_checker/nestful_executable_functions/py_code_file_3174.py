from random import randint

import random
random.seed(42)
from typing import List



def generate_midi_note() -> int:

    """Generates a random MIDI note number between 53 and 72."""

    return randint(53, 72)



def generate_musical_sequence(steps: int) -> List[int]:

    """

    Generates a musical sequence of MIDI note numbers by randomly choosing notes from the range [53, 72].

    The sequence should be composed of 20 notes. The interval between consecutive notes is determined by the

    `steps` parameter.



    Args:

        steps: The interval between consecutive notes.



    Returns:

        A list of 20 MIDI note numbers.

    """

    midi_notes = [generate_midi_note() for _ in range(20)]

    musical_sequence = [midi_notes[0]] + [midi_notes[i] + steps for i in range(1, 20)]

    return musical_sequence

