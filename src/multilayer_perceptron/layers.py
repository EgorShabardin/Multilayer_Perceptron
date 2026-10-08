from abc import ABC, abstractmethod
import numpy as np

from .activation_functions import get_activation, AbstractActivationFunction
from .initializers import get_initialization


class AbstractLayer(ABC):
    def __init__(self) -> None:
        self._parameters: dict[str, np.ndarray] = {}
        self._gradients: dict[str, np.ndarray] = {}
    
    @abstractmethod
    def forward(self, input_matrix: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        pass

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        return self._parameters

    @property
    def gradients(self) -> dict[str, np.ndarray]:
        return self._gradients

class DenseLayer(AbstractLayer):
    def __init__(self, in_features: int, out_features: int, activation_name: str, initialization_name: str, 
                  rng: np.random.Generator | None = None, activation_kwargs: dict | None = None) -> None:
        
        super().__init__()
        self._in_features: int = in_features
        self._out_features: int = out_features
        self._input_matrix: np.ndarray | None = None
        self._activation_name = activation_name
        self._initialization_name = initialization_name
        self._activation_kwargs = activation_kwargs

        self._activation_function: AbstractActivationFunction = get_activation(activation_name, **(activation_kwargs or {}))
        self._parameters = get_initialization(initialization_name, in_features, out_features).init_parameters(rng)

    def forward(self, input_matrix: np.ndarray) -> np.ndarray:
        input_matrix = np.asarray(input_matrix, dtype=float)

        if input_matrix.ndim != 2 or input_matrix.shape[1] != self._in_features:
            raise ValueError(f"Expected input of shape (batch, {self._in_features}), got {input_matrix.shape}")
    
        self._input_matrix = input_matrix
        return self._activation_function.forward(input_matrix @ self._parameters["weight"] + self._parameters["bias"])

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        if self._input_matrix is None:
            raise RuntimeError("backward() was called before forward()")
        
        delta: np.ndarray = self._activation_function.backward(output_gradient)
        self._gradients["weight"] = self._input_matrix.T @ delta
        self._gradients["bias"] = np.sum(delta, axis=0, keepdims=True)

        return delta @ self._parameters["weight"].T

    @property
    def config(self) -> dict:
        return {"in_features": self._in_features, "out_features": self._out_features,
                "activation_name": self._activation_name, "initialization_name": self._initialization_name,
                "activation_kwargs": self._activation_kwargs}

    def set_parameters(self, parameters: dict[str, np.ndarray]) -> None:
        for name, value in parameters.items():
            value = np.asarray(value, dtype=float)

            if name not in self._parameters or value.shape != self._parameters[name].shape:
                raise ValueError(f"Parameter {name} has wrong name or shape {value.shape}")

            self._parameters[name] = value.copy()