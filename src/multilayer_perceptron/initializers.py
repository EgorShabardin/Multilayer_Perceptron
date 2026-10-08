from abc import ABC, abstractmethod
import numpy as np

from .registry import Registry


_INITIALIZATIONS_REGISTRY: Registry = Registry("initializers")

class AbstractInitializer(ABC):
    def __init__(self, in_features: int, out_features: int) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive integers")

        self._in_features: int = in_features
        self._out_features: int = out_features

    def __init_subclass__(cls, initialization_name: str | None = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        initialization_name = initialization_name if initialization_name is not None else cls.__name__
        _INITIALIZATIONS_REGISTRY.register(initialization_name, cls)

    def init_parameters(self, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
        rng = rng if rng is not None else np.random.default_rng()

        return {"weight": self._init_weight(rng), "bias": np.zeros((1, self._out_features), dtype=float)}

    @abstractmethod
    def _init_weight(self, rng: np.random.Generator) -> np.ndarray:
        pass

class XavierNormal(AbstractInitializer, initialization_name="xavier_normal"):
    def _init_weight(self, rng: np.random.Generator) -> np.ndarray:
        std: float = np.sqrt(2.0 / (self._in_features + self._out_features))
        return rng.normal(loc=0.0, scale=std, size=(self._in_features, self._out_features))

class XavierUniform(AbstractInitializer, initialization_name="xavier_uniform"):
    def _init_weight(self, rng: np.random.Generator) -> np.ndarray:
        limit: float = np.sqrt(6.0 / (self._in_features + self._out_features))
        return rng.uniform(low=-limit, high=limit, size=(self._in_features, self._out_features))

class KaimingNormal(AbstractInitializer, initialization_name="kaiming_normal"):
    def _init_weight(self, rng: np.random.Generator) -> np.ndarray:
        std: float = np.sqrt(2.0 / self._in_features)
        return rng.normal(loc=0.0, scale=std, size=(self._in_features, self._out_features))

class KaimingUniform(AbstractInitializer, initialization_name="kaiming_uniform"):
    def _init_weight(self, rng: np.random.Generator) -> np.ndarray:
        limit: float = np.sqrt(6.0 / self._in_features)
        return rng.uniform(low=-limit, high=limit, size=(self._in_features, self._out_features))

def get_initialization(initialization_name: str, in_features: int,
                       out_features: int, *args, **kwargs) -> AbstractInitializer:
    return _INITIALIZATIONS_REGISTRY.create(initialization_name, in_features, out_features, *args, **kwargs)