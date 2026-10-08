from abc import ABC, abstractmethod

from .layers import AbstractLayer
from .registry import Registry


_OPTIMIZERS_REGISTRY: Registry = Registry("optimizers")

class AbstractOptimizer(ABC):
    def __init_subclass__(cls, optimizer_name: str | None = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        optimizer_name = optimizer_name if optimizer_name is not None else cls.__name__
        _OPTIMIZERS_REGISTRY.register(optimizer_name, cls)

    @abstractmethod
    def update(self, layers: list[AbstractLayer]) -> None:
        pass

class StochasticGradientDescent(AbstractOptimizer, optimizer_name="sgd"):
    def __init__(self, learning_rate: float = 1e-3):
        self._learning_rate = learning_rate

    def update(self, layers: list[AbstractLayer]) -> None:
        for layer in layers:
            for name, parameter in layer.parameters.items():
                parameter -= self._learning_rate * layer.gradients[name]

def get_optimizer(optimizer_name: str, *args, **kwargs) -> AbstractOptimizer:
    return _OPTIMIZERS_REGISTRY.create(optimizer_name, *args, **kwargs)