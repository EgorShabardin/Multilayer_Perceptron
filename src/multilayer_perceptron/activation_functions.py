from abc import ABC, abstractmethod
import numpy as np

from .registry import Registry


_ACTIVATION_REGISTRY: Registry = Registry("activations")

class AbstractActivationFunction(ABC):
    def __init__(self) -> None:
        self._cache: np.ndarray | None = None
        self._shape: tuple[int, ...] | None = None

    def __init_subclass__(cls, activation_name: str | None = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        activation_name = activation_name if activation_name is not None else cls.__name__
        _ACTIVATION_REGISTRY.register(activation_name, cls)

    @abstractmethod
    def _forward(self, input_matrix: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _backward(self, output_gradient: np.ndarray) -> np.ndarray:
        pass

    def forward(self, input_matrix: np.ndarray) -> np.ndarray:
        input_matrix = np.asarray(input_matrix, dtype=float)
        self._shape = input_matrix.shape
        return self._forward(input_matrix)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        if self._shape is None:
            raise RuntimeError("backward() was called before forward()")

        output_gradient = np.asarray(output_gradient, dtype=float)

        if output_gradient.shape != self._shape:
            raise ValueError(f"Shape mismatch: expected {self._shape}, got {output_gradient.shape}")

        return self._backward(output_gradient)

class Identity(AbstractActivationFunction, activation_name="identity"):
    def _forward(self, input_matrix: np.ndarray) -> np.ndarray:
        return input_matrix.copy()

    def _backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient.copy()

class ReLU(AbstractActivationFunction, activation_name="relu"):
    def _forward(self, input_matrix: np.ndarray) -> np.ndarray:
        self._cache = input_matrix > 0.0
        return np.maximum(input_matrix, 0.0)

    def _backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return np.where(self._cache, output_gradient, 0.0)

class LeakyReLU(AbstractActivationFunction, activation_name="leaky_relu"):
    def __init__(self, slope: float = 0.01) -> None:
        if slope <= 0.0 or slope >= 1.0:
            raise ValueError("Slope must be greater than zero and less than one")

        super().__init__()
        self._slope: float = slope

    def _forward(self, input_matrix: np.ndarray) -> np.ndarray:
        self._cache = input_matrix > 0.0
        return np.maximum(input_matrix, self._slope * input_matrix)

    def _backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return np.where(self._cache, output_gradient, self._slope * output_gradient)

class Sigmoid(AbstractActivationFunction, activation_name="sigmoid"):
    def _forward(self, input_matrix: np.ndarray) -> np.ndarray:
        self._cache = 1.0 / (1.0 + np.exp(-np.clip(input_matrix, -500, 500)))
        return self._cache.copy()

    def _backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * self._cache * (1.0 - self._cache)

class Tanh(AbstractActivationFunction, activation_name="tanh"):
    def _forward(self, input_matrix: np.ndarray) -> np.ndarray:
        self._cache = np.tanh(input_matrix)
        return self._cache.copy()

    def _backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * (1.0 - self._cache * self._cache)

class Softmax(AbstractActivationFunction, activation_name="softmax"):
    def _forward(self, input_matrix: np.ndarray) -> np.ndarray:
        exp_value = np.exp(input_matrix - np.max(input_matrix, axis=-1, keepdims=True))
        self._cache = exp_value / np.sum(exp_value, axis=-1, keepdims=True)
        return self._cache.copy()

    def _backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return self._cache * (output_gradient - np.sum(output_gradient * self._cache, axis=-1, keepdims=True))

def get_activation(activation_name: str, *args, **kwargs) -> AbstractActivationFunction:
    return _ACTIVATION_REGISTRY.create(activation_name, *args, **kwargs)