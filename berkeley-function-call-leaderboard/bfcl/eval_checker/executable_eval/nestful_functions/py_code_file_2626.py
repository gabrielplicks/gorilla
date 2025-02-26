from typing import List



def calculate_scale_degree(note: str, scale: List[str]) -> int:

    """Calculates the scale degree of a note in a given music scale.



    Args:

        note: The note to be mapped to the scale.

        scale: The scale (sequence of notes).



    Returns:

        The scale degree of the note in the given scale. If the note is not present in the scale, returns -1.

    """

    if note not in scale:

        return -1

    scale_degree_mapping = {scale_note: i + 1 for i, scale_note in enumerate(scale)}

    return scale_degree_mapping[note]

