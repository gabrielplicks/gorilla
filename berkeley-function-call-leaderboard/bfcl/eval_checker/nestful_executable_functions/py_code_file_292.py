from typing import List, Tuple



def filter_positive_and_negative_examples(examples: List[Tuple[Any, int]]) -> Tuple[List[Any], List[Any]]:

    """Filters a list of examples into positive and negative examples.



    Args:

        examples: A list of tuples, where each tuple contains an example and its label.



    Returns:

        A tuple containing two lists: one for positive examples and the other for negative examples.

    """

    positive_examples = []

    negative_examples = []

    for example, label in examples:

        if label == 0:

            negative_examples.append(example)

        elif label == 1:

            positive_examples.append(example)

    return (positive_examples, negative_examples)

