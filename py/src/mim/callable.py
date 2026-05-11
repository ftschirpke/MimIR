from typing import Any


class MimCallable:

    def __init__(self, name: str, input_types, return_type, function_ptr):
        self.name = name
        self.input_types = input_types
        self.return_types = return_type
        self.function_ptr = function_ptr
        self.function_ptr.argtypes = input_types
        self.function_ptr.restype = return_type

    def __call__(self, *args: Any) -> Any:

        return self.function_ptr(*args)
