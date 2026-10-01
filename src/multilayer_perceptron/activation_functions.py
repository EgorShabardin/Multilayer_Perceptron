from abc import ABC, abstractmethod
import numpy as np


class AbstractActivation(ABC):
    def __init__(self) -> None:
        self._cache: np.ndarray | None = None
        self._shape: tuple[int, ...] | None = None

    def __init_subclass__(cls, activation_name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)

        if activation_name is not None:
            activation_name = activation_name.lower()

            if activation_name in _ACTIVATIONS:
                raise ValueError(f"Class with name {activation_name} exists")
   
            _ACTIVATIONS[activation_name] = cls

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
        if self._cache is None:
            raise RuntimeError("backward() was called before forward()")

        output_gradient = np.asarray(output_gradient, dtype=float)

        if output_gradient.shape != self._shape:
            raise ValueError(f"Gradient dimension mismatch: expected {self._shape}, got {output_gradient.shape}")

        return self._backward(output_gradient)

    def cache_reset(self) -> None:
        self._cache = None
        self._shape = None

_ACTIVATIONS: dict[str, type[AbstractActivation]] = dict()

class Identity(AbstractActivation, activation_name="identity"):
    def _forward(self, input_matrix: np.ndarray) -> np.ndarray:
        self._cache = input_matrix
        return input_matrix

    def _backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient

class ReLU(AbstractActivation, activation_name="relu"):
    def _forward(self, input_matrix: np.ndarray) -> np.ndarray:
        self._cache = input_matrix > 0.0
        return np.maximum(input_matrix, 0.0)

    def _backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return np.where(self._cache, output_gradient, 0.0)

class LeakyReLU(AbstractActivation, activation_name="leaky_relu"):
    def __init__(self, negative_slope: float = 0.01) -> None:
        if negative_slope <= 0.0 or negative_slope >= 1.0:
            raise ValueError("Negative_slope must be greater than zero and less than one")

        super().__init__()
        self._negative_slope: float = negative_slope

    def _forward(self, input_matrix: np.ndarray) -> np.ndarray:
        self._cache = input_matrix > 0.0
        return np.maximum(input_matrix, self._negative_slope * input_matrix)

    def _backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return np.where(self._cache, output_gradient, self._negative_slope * output_gradient)

class Sigmoid(AbstractActivation, activation_name="sigmoid"):
    def _forward(self, input_matrix: np.ndarray) -> np.ndarray:
        self._cache = 1.0 / (1.0 + np.exp(-np.clip(input_matrix, -500, 500)))
        return self._cache

    def _backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * self._cache * (1 - self._cache)

class Tanh(AbstractActivation, activation_name="tanh"):
    def _forward(self, input_matrix: np.ndarray) -> np.ndarray:
        self._cache = np.tanh(input_matrix)
        return self._cache

    def _backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * (1 - self._cache * self._cache)

class Softmax(AbstractActivation, activation_name="softmax"):
    def _forward(self, input_matrix: np.ndarray) -> np.ndarray:
        exp_value: np.ndarray = np.exp(input_matrix - np.max(input_matrix, axis=-1, keepdims=True))
        self._cache = exp_value / np.sum(exp_value, axis=-1, keepdims=True)
        return self._cache

    def _backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return self._cache * (output_gradient - np.sum(output_gradient * self._cache, axis=-1, keepdims=True))

def get_activation(activation_name: str, **kwargs) -> AbstractActivation:
    activation_name = activation_name.strip().lower().replace(" ", "_").replace("-", "_")

    if activation_name not in _ACTIVATIONS:
        raise ValueError(f"Class with name {activation_name} does not exist")

    return _ACTIVATIONS[activation_name](**kwargs)