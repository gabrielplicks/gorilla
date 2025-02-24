from typing import Dict, List



def contains_utr(transcript: Dict[str, List[Dict[str, int]]], start_pos: int, end_pos: int) -> bool:

    """Determines if there is at least one UTR exon contained within the given range.



    Args:

        transcript: A dictionary containing exons and UTR information.

        start_pos: The start position of the range.

        end_pos: The end position of the range.



    Returns:

        A boolean value indicating whether there is at least one UTR exon contained within the given range.

    """

    for exon in transcript["exons"]:

        if exon["type"] == "UTR":

            if start_pos >= exon["start"] and end_pos <= exon["end"]:

                return True

    return False

