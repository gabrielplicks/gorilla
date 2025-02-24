from typing import TextIO



def count_pattern_in_fasta(fasta_file: TextIO, pattern: str) -> int:

    """Counts the number of times a DNA sequence pattern appears in a FASTA file.



    Args:

        fasta_file: A file containing DNA sequences in the FASTA format.

        pattern: The DNA sequence pattern to search for.



    Returns:

        The number of times the pattern appears in the sequences in the file.

    """

    count = 0

    with open(fasta_file, 'r') as f:

        for line in f:

            line = line.strip()

            if line.startswith('>'):

                continue

            count += line.count(pattern)

    return count

