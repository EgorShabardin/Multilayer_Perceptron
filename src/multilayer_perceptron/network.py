import numpy as np
import json

from .layers import DenseLayer
from .loss_functions import get_loss_function
from .activation_functions import get_activation
from .optimizers import get_optimizer


class Network:
    def __init__(self, network_layers: list[DenseLayer], lossfunction_name: str, optimizer_name: str,
                 learning_rate: float, rng: np.random.Generator | None = None):
        
        self._network_layers = network_layers
        self._loss_function = get_loss_function(lossfunction_name)
        self._optimizer = get_optimizer(optimizer_name, learning_rate)
        self._rng = rng if rng is not None else np.random.default_rng()
        self._lossfunction_name = lossfunction_name
        self._optimizer_name = optimizer_name
        self._learning_rate = learning_rate

    def _train_step(self, input_matrix: np.ndarray, target: np.ndarray) -> None:
        output_matrix = self.predict(input_matrix)
        self._loss_function.forward(output_matrix, target)
        output_gradient = self._loss_function.backward()

        for layer in reversed(self._network_layers):
            output_gradient = layer.backward(output_gradient)
    
        self._optimizer.update(self._network_layers)

    def predict(self, input_matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(input_matrix, dtype=float)
    
        for layer in self._network_layers:
            matrix = layer.forward(matrix)
            
        return matrix

    def evaluate(self, input_matrix: np.ndarray, target: np.ndarray) -> tuple[float, float]:
        output_matrix = self.predict(input_matrix)
        loss = self._loss_function.forward(output_matrix, np.asarray(target, dtype=float))
        accuracy = float((output_matrix.argmax(axis=1) == np.asarray(target).argmax(axis=1)).mean())
        return loss, accuracy

    def fit(self, input_matrix: np.ndarray, target: np.ndarray, epoch: int, batch_size: int | None = None,
            validation: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, list[float]]:

        input_matrix = np.asarray(input_matrix, dtype=float)
        target = np.asarray(target, dtype=float)
        batch_size = batch_size if batch_size is not None else len(input_matrix)
    
        history: dict[str, list[float]] = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    
        for i in range(epoch):
            order = self._rng.permutation(len(input_matrix))

            for start in range(0, len(input_matrix), batch_size):
                batch = order[start:start + batch_size]
                self._train_step(input_matrix[batch], target[batch])
    
            loss, acc = self.evaluate(input_matrix, target)
            history["loss"].append(loss)
            history["acc"].append(acc)
            message = f"epoch {i + 1}/{epoch} - loss: {loss:.4f}"
    
            if validation is not None:
                val_loss, val_acc = self.evaluate(*validation)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                message += f" - val_loss: {val_loss:.4f}"
    
            print(message)
        return history

    def save(self, path: str, **extra: np.ndarray) -> None:
        config = {"layers": [layer.config for layer in self._network_layers],
                  "lossfunction_name": self._lossfunction_name,
                  "optimizer_name": self._optimizer_name,
                  "learning_rate": self._learning_rate}
 
        arrays: dict[str, np.ndarray] = {"config": np.array(json.dumps(config))}
 
        for i, layer in enumerate(self._network_layers):
            for name, parameter in layer.parameters.items():
                arrays[f"layer{i}_{name}"] = parameter
 
        for name, value in extra.items():
            if name == "config" or name.startswith("layer"):
                raise ValueError(f"Name {name} is reserved")
            arrays[f"extra_{name}"] = np.asarray(value)
 
        np.savez(path, **arrays)
 
    @classmethod
    def load(cls, path: str) -> tuple["Network", dict[str, np.ndarray]]:
        with np.load(path, allow_pickle=False) as data:
            config = json.loads(str(data["config"]))
            layers: list[DenseLayer] = []
 
            for i, layer_config in enumerate(config["layers"]):
                layer = DenseLayer(**layer_config)
                layer.set_parameters({name: data[f"layer{i}_{name}"] for name in layer.parameters})
                layers.append(layer)
 
            extra = {key[len("extra_"):]: data[key] for key in data.files if key.startswith("extra_")}
 
        network = cls(layers, config["lossfunction_name"], config["optimizer_name"], config["learning_rate"])
        return network, extra