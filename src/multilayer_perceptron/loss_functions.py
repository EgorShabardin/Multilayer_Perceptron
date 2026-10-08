from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from .registry import Registry


_LOSSFUNCTIONS_REGISTRY: Registry = Registry("losses")

class AbstractLossFunction(ABC):
    def __init__(self) -> None:
        self._cache: dict[str, Any] | None = None

    def __init_subclass__(cls, lossfunction_name: str | None = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        lossfunction_name = lossfunction_name if lossfunction_name is not None else cls.__name__
        _LOSSFUNCTIONS_REGISTRY.register(lossfunction_name, cls)

    @abstractmethod
    def _forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        pass

    @abstractmethod
    def _backward(self) -> np.ndarray:
        pass

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        prediction = np.asarray(prediction, dtype=float)
        target = np.asarray(target, dtype=float)

        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch: prediction {prediction.shape}, target {target.shape}")

        return self._forward(prediction, target)

    def backward(self) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("backward() was called before forward()")

        return self._backward()

class MeanSquaredError(AbstractLossFunction, lossfunction_name="mse"):
    def _forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        difference: np.ndarray = prediction - target
        self._cache = {"difference": difference, "size": prediction.size}
        return float(np.mean(difference * difference))

    def _backward(self) -> np.ndarray:
        difference: np.ndarray = self._cache["difference"]
        size: int = self._cache["size"]
        return 2.0 * difference / size

class MeanAbsoluteError(AbstractLossFunction, lossfunction_name="mae"):
    def _forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        difference: np.ndarray = prediction - target
        self._cache = {"difference": difference, "size": prediction.size}
        return float(np.mean(np.abs(difference)))

    def _backward(self) -> np.ndarray:
        difference: np.ndarray = self._cache["difference"]
        size: int = self._cache["size"]
        return np.sign(difference) / size

class BinaryCrossEntropy(AbstractLossFunction, lossfunction_name="binary_cross_entropy"):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self._eps: float = eps

    def _forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        prediction = np.clip(prediction, self._eps, 1.0 - self._eps)
        self._cache = {"prediction": prediction, "target": target}
        return float(np.mean(-(target * np.log(prediction) + (1.0 - target) * np.log(1.0 - prediction))))

    def _backward(self) -> np.ndarray:
        prediction: np.ndarray = self._cache["prediction"]
        target: np.ndarray = self._cache["target"]
        return ((prediction - target) / (prediction * (1.0 - prediction))) / prediction.size

class CategoricalCrossEntropy(AbstractLossFunction, lossfunction_name="categorical_cross_entropy"):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self._eps: float = eps
    
    def _forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        prediction = np.clip(prediction, self._eps, 1.0 - self._eps)
        self._cache = {"prediction": prediction, "target": target}
        return float(-np.mean(np.sum(target * np.log(prediction), axis=-1)))

    def _backward(self) -> np.ndarray:
        prediction: np.ndarray = self._cache["prediction"]
        target: np.ndarray = self._cache["target"]
        n_samples: int = prediction.size // prediction.shape[-1]
        return -target / prediction / n_samples

def get_loss_function(lossfunction_name: str, *args, **kwargs) -> AbstractLossFunction:
    return _LOSSFUNCTIONS_REGISTRY.create(lossfunction_name, *args, **kwargs)