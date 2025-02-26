import os

from typing import Dict



def generate_word_frequency_dictionary() -> Dict[str, int]:

    """Generates a dictionary from a text file of words, where each line represents a word and its corresponding frequency of occurrence.

    The file is located in a subdirectory named `input` with file name `src-freq.txt`.

    The dictionary keys are words and the values are their frequencies.



    Returns:

        A dictionary where the keys are words and the values are their frequencies.

    """

    file_path = os.path.join('input', 'src-freq.txt')

    word_frequency_dict = {}

    with open(file_path, 'r') as file:

        for line in file:

            word, frequency = line.strip().split(' ')

            word_frequency_dict[word] = int(frequency)



    return word_frequency_dict

