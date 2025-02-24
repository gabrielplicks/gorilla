import pandas as pd

from typing import Any



def is_pandas_object(obj: Any) -> bool:

    """Determines whether a given object is a pandas DataFrame, Series, or one of their subclasses.



    Args:

        obj: The object to check.



    Returns:

        A boolean value indicating whether the object is a pandas DataFrame, Series, or one of their subclasses.

    """

    return isinstance(obj, (pd.DataFrame, pd.Series)) or \

        any(isinstance(obj, cls) for cls in pd.core.base.PandasObject.__subclasses__())

