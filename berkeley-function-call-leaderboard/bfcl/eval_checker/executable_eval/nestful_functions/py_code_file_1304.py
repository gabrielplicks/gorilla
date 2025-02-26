import types



def monkey_patch_class(cls: type, attr_name: str, attr_value: any):

    """Applies monkey patching to a given class, adding a new class attribute and method.



    Args:

        cls: The class to be patched.

        attr_name: The name of the new attribute.

        attr_value: The value of the attribute.

    """

    def dummy():

        pass

    method = types.MethodType(dummy, cls)

    setattr(cls, attr_name, method)

    setattr(cls, attr_name, attr_value)

