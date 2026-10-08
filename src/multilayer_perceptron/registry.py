from typing import Any


class Registry:
    def __init__(self, registry_type: str) -> None:
        self._registry_type: str = registry_type
        self._registry_dict: dict[str, type] = {}

    @staticmethod
    def _normalize_name(registration_name: str) -> str:
        return registration_name.strip().lower().replace(" ", "_").replace("-", "_")

    def register(self, registration_name: str, cls: type) -> None:
        registration_name = self._normalize_name(registration_name)

        if registration_name in self._registry_dict:
            raise ValueError(f"Class with name {registration_name} is already registered in {self._registry_type}")
        
        self._registry_dict[registration_name] = cls

    def create(self, registration_name: str, *args, **kwargs) -> Any:
        registration_name = self._normalize_name(registration_name)

        if registration_name not in self._registry_dict:
            raise ValueError(f"Class with name {registration_name} is not registered in {self._registry_type}")

        return self._registry_dict[registration_name](*args, **kwargs)