from typing import Tuple



def make_sentence(person: str, emotion: str, color: str, object: str) -> str:

    """Constructs a sentence in the format: "The {color} {object} makes {person} {emotion}."



    Args:

        person: The subject of the sentence.

        emotion: The emotion of the subject.

        color: The color of the object.

        object: The object of the sentence.



    Returns:

        A sentence in the format: "The {color} {object} makes {person} {emotion}."

    """

    sentence = f"The {color} {object} makes {person} {emotion}."

    return sentence

