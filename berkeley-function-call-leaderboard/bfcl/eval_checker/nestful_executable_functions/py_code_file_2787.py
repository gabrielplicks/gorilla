from typing import Optional



class Foo:

    def __init__(self, bar: int, baz: int):

        self.bar = bar

        self.baz = baz



def create_foo_instance(bar: Optional[int] = 0, baz: Optional[int] = 1) -> Foo:

    """Creates an object instance of Foo with the desired attributes.



    Args:

        bar: The value to set for the bar attribute of the Foo object. Default is 0.

        baz: The value to set for the baz attribute of the Foo object. Default is 1.



    Returns:

        An object instance of Foo with the specified attributes.

    """

    return Foo(bar, baz)

