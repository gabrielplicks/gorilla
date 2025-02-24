from typing import List



def find_num_sentences(sentences: List[str]) -> int:

    """Finds the number of sentences whose tokens cover at least 80% of the total number of tokens.



    Args:

        sentences: A list of strings, where each string represents a sentence.



    Returns:

        An integer representing the number of sentences whose tokens cover at least 80% of the total number of tokens.

    """

    total_tokens = sum(len(sentence.split()) for sentence in sentences)

    sorted_sentences = sorted(sentences, key=lambda sentence: len(sentence.split()))

    running_total = 0

    num_sentences = 0

    for sentence in sorted_sentences:

        num_tokens = len(sentence.split())

        if running_total + num_tokens > 0.8 * total_tokens:

            break

        running_total += num_tokens

        num_sentences += 1

    return num_sentences

