import csv

from typing import Dict



def count_words_in_csv(filename: str) -> Dict[str, int]:

    """

    Counts the number of occurrences of each word in a CSV file.



    Args:

        filename: The name of the CSV file.



    Returns:

        A dictionary with the words as keys and the number of occurrences as values.

    """

    word_counts = {}

    with open(filename, 'r') as csvfile:

        csvreader = csv.reader(csvfile)

        for line in csvreader:

            for word in line:

                word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts

