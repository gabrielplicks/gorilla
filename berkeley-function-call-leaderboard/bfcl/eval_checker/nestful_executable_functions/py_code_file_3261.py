from typing import List, Tuple



def validate_exons(exons: List[Tuple[int, int]]) -> bool:

    """Validate the exon definitions of a gene.



    Args:

        exons: A list of tuples representing the exon definitions. Each tuple contains the start and end positions of an exon.



    Returns:

        True if the exon definitions are valid, False otherwise.

    """

    for i in range(len(exons) - 1):

        if exons[i + 1][0] < exons[i][1]:

            return False

    return len(exons) == len(set(exons))

