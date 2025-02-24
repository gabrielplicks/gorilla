import math



def compute_entropy(nucleotide_frequencies: list[float]) -> float:

    """Calculates the Shannon entropy of a sequence of nucleotides.



    Args:

        nucleotide_frequencies: A list of nucleotide frequencies.



    Returns:

        The Shannon entropy as a floating-point number.

    """

    entropy = 0



    for frequency in nucleotide_frequencies:

        if frequency > 0:

            entropy += frequency * math.log2(frequency)



    return -entropy

