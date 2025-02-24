from typing import Union, Callable



def compute_learning_rate(x: Union[float, None]) -> Union[float, None]:

    """Computes a learning rate given a normalized value x.



    Args:

        x: A normalized value.



    Returns:

        The learning rate.

    """

    switch_funcs = {

        None: lambda x: None,

        0.05: lambda x: 0.05,

        0.1: lambda x: 0.1,

        lambda x: x < 0.05: lambda x: 0.05,

        lambda x: x > 0.1: lambda x: 0.1,

    }



    for case, func in switch_funcs.items():

        if callable(case) and case(x):

            return func(x)

        elif case == x:

            return func(x)

