from typing import Any


def install(module_globals: dict[str, Any], name: str, exported: Any) -> Any:
    module_globals[name] = exported

    def __getattr__(attr: str) -> Any:
        return getattr(exported, attr)

    def __dir__() -> list[str]:
        return sorted(set(module_globals) | set(dir(exported)))

    module_globals["__getattr__"] = __getattr__
    module_globals["__dir__"] = __dir__
    return exported
