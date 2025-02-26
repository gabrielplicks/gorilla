import re

from typing import List, Tuple



def extract_concept_from_text(text: str) -> List[Tuple[str, str]]:

    """Extracts concepts from a given text string.



    Args:

        text: The input text string.



    Returns:

        A list of concept tuples, where each tuple consists of the concept name and its category.

    """

    concepts: List[Tuple[str, str]] = []

    matches = re.findall(r'\(([^)]+)\)', text)

    for match in matches:

        name, _, category = match.partition(':')

        concepts.append((name, category or None))

    return concepts

