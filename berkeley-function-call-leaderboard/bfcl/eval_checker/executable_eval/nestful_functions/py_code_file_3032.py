from typing import List



class FeatureObject:

    def __init__(self, feature: str, value):

        self.feature = feature

        self.value = value



def extract_feature_values(feat_obj_pairs: List[FeatureObject], feature: str) -> List[str]:

    """Extracts the feature values of a specified feature from a list of feature-object pairs.

    Args:

        feat_obj_pairs: A list of feature-object pairs.

        feature: The specified feature name.

    """

    feature_values = []

    for feat_obj_pair in feat_obj_pairs:

        if feat_obj_pair.feature == feature:

            feature_values.append(feat_obj_pair.value)

    return feature_values

