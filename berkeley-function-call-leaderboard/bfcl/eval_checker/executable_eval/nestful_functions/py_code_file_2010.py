import random

random.seed(42)
from typing import Dict, List



def generate_random_features(features: Dict[str, List[str]]) -> Dict[str, str]:

    """Generates random feature values for a given set of features.



    Args:

        features: A dictionary of features, where each key represents a feature name and each value represents a list of possible feature values.



    Returns:

        A dictionary of features with randomly selected values.

    """

    for key, value in features.items():

        features[key] = random.choice(value)

    return features

