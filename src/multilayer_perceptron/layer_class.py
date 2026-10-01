from collections.abc import Callable
from abc import ABC, abstractmethod
import numpy as np


class AbstractLayer (ABC):
    def __init__ (self, in_features: int, out_features: int, activation_name: str, initialization_name: str) -> None:
        self._in_features: int = in_features
        self._out_features: int = out_features

        self._activation_name: str = activation_name.lower()
        self._initialization_name: str = initialization_name.lower()

        self._parameters: dict[str, np.ndarray] = {}
        self._gradients: dict[str, np.ndarray] = {}

        self._input_data: np.ndarray | None = None
        self._linear_data: np.ndarray | None = None
        self._activate_data: np.ndarray | None = None

        self._activation: Callable[[np.ndarray], np.ndarray] | None = None
        self._grad_activation: Callable[[np.ndarray], np.ndarray] | None = None

        self._parameters_init()
        self._activation_init()
        
    @abstractmethod
    def _parameters_init (self) -> None:
        pass

    @abstractmethod
    def _activation_init (self) -> None:
        pass

    @abstractmethod
    def forward_propagation (self, data_matrix: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def back_propagation (self, data_matrix: np.ndarray) -> np.ndarray:
        pass

    @property
    def parameters (self) -> dict[str, np.ndarray]:
        return self._parameters

    @property
    def gradients (self) -> dict[str, np.ndarray]:
        return self._gradients


class DenseLayer (AbstractLayer):
    def __init__ (self, in_features: int, out_features: int, activation_name: str, initialization_name: str, leaky_rate: float = 0.01) -> None:
        self.__leaky_rate = leaky_rate
        super().__init__(in_features, out_features, activation_name, initialization_name)

    def _parameters_init (self) -> None:
        self._parameters["bias"] = np.zeros((1, self._out_features), dtype=float)

        if self._initialization_name == "xavier_normal":
            std: float = np.sqrt(2.0 / (self._in_features + self._out_features))
            self._parameters["weight"] = np.random.normal(loc=0.0, scale=std, size=(self._in_features, self._out_features))

        elif self._initialization_name == "xavier_uniform":
            limit: float = np.sqrt(6.0 / (self._in_features + self._out_features))
            self._parameters["weight"] = np.random.uniform(low=-limit, high=limit, size=(self._in_features, self._out_features))

        elif self._initialization_name == "kaiming_normal":
            std: float = np.sqrt(2.0 / self._in_features)
            self._parameters["weight"] = np.random.normal(loc=0.0, scale=std, size=(self._in_features, self._out_features))
        
        elif self._initialization_name == "kaiming_uniform":
            limit: float = np.sqrt(6.0 / self._in_features)
            self._parameters["weight"] = np.random.uniform(low=-limit, high=limit, size=(self._in_features, self._out_features))

        else:
            raise ValueError(f"Unknown initialization method: '{self._initialization_name}'")

    def _activation_init (self) -> None:
        if self._activation_name == "sigmoid":
            self._activation = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
            self._grad_activation = lambda x: self._activation(x) * (1 - self._activation(x))

        elif self._activation_name == "tanh":
            self._activation = lambda x: np.tanh(x)
            self._grad_activation = lambda x: 1 - self._activation(x) ** 2

        elif self._activation_name == "relu":
            self._activation = lambda x: np.where(x > 0.0, x, 0.0)
            self._grad_activation = lambda x: np.where(x > 0.0, 1.0, 0.0)

        elif self._activation_name == "leaky_relu":
            self._activation = lambda x: np.where(x > 0.0, x, self.__leaky_rate * x)
            self._grad_activation = lambda x: np.where(x > 0.0, 1.0, self.__leaky_rate)

        elif self._activation_name == "softmax":
            self._activation = lambda x: np.exp(x - np.max(x, axis=1, keepdims=True)) / \
                np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True)
            self._grad_activation = None

        else:
            raise ValueError(f"Unknown activation function: '{self._activation_name}'")

    def forward_propagation (self, data_matrix: np.ndarray) -> np.ndarray:
        self._input_data = data_matrix
        self._linear_data = data_matrix @ self._parameters["weight"] + self._parameters["bias"]
        self._activate_data = self._activation(self._linear_data)

        return self._activate_data

    def back_propagation (self, data_matrix: np.ndarray) -> np.ndarray:
        if self._input_data is None:
            raise RuntimeError("Back propagation called before forward propagation")

        delta: np.ndarray = data_matrix * self._grad_activation(self._linear_data)
        self._gradients["weight"] = self._input_data.T @ delta
        self._gradients["bias"] = np.sum(delta, axis=0, keepdims=True)

        return delta @ self._parameters["weight"].T
