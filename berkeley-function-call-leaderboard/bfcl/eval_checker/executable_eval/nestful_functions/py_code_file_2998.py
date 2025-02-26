import pandas as pd

from typing import Dict



def select_top_features(feature_importance: pd.Series, num_features: int) -> Dict[int, float]:

    """Selects the top `num_features` features from a Pandas series `feature_importance`.



    Args:

        feature_importance: A Pandas series containing feature importances.

        num_features: The number of top features to select.



    Returns:

        A dictionary where the keys are the indices of the top features and the values are their importances.

    """

    top_features = []

    sorted_indices = feature_importance.sort_values(ascending=False).index

    for i, feature_index in enumerate(sorted_indices):

        if i < num_features:

            top_features.append(feature_index)

    selected_features = {}

    for feature_index in top_features:

        selected_features[feature_index] = feature_importance[feature_index]

    return selected_features

