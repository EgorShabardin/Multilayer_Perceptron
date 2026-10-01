import numpy as np


class MultilayerPerceptron:
    def __init__ (self, layers: int, leak_rate: float = 0.01, learning_rate: float = 0.01, epsilon: float = 1e-5):
        self.__learning_rate = learning_rate
        self.__leak_rate = leak_rate
        self.__epsilon = epsilon
        self.__layers = layers

    def __weight_init (self, X: np.ndarray, Y: np.ndarray):
        hidden_limit = np.sqrt(2.0 / X.shape[1])
        self.__hidden_weights = np.random.normal(loc=0.0, scale=hidden_limit, size=(self.__layers, X.shape[1], X.shape[1]))
        self.__hidden_bias = np.zeros((self.__layers, 1, X.shape[1]), dtype=float)

        output_limit = np.sqrt(2.0 / (X.shape[1] + Y.shape[1]))
        self.__output_weights = np.random.normal(loc=0.0, scale=output_limit, size=(X.shape[1], Y.shape[1]))
        self.__output_bias = np.zeros((1, Y.shape[1]), dtype=float)

    def __leaky_relu (self, layer_matrix: np.ndarray):
        return np.where(layer_matrix >= 0.0, layer_matrix, self.__leak_rate * layer_matrix)

    def __leaky_relu_grad (self, layer_matrix: np.ndarray):
        return np.where(layer_matrix >= 0.0, 1.0, self.__leak_rate)

    def __softmax (self, layer_matrix: np.ndarray):
        exponents_values = np.exp(layer_matrix - np.max(layer_matrix, axis=1, keepdims=True))
        return exponents_values / np.sum(exponents_values, axis=1, keepdims=True)

    def __forward_propagation (self, X: np.ndarray):
        output_matrix = X
        layers_matrices_lst = [(None, output_matrix)]
    
        for i in range(self.__layers):
            input_matrix = output_matrix @ self.__hidden_weights[i] + self.__hidden_bias[i]
            output_matrix = self.__leaky_relu(input_matrix)

            layers_matrices_lst.append((input_matrix, output_matrix))

        input_matrix = output_matrix @ self.__output_weights + self.__output_bias
        output_matrix = self.__softmax(input_matrix)
        layers_matrices_lst.append((input_matrix, output_matrix))
    
        return layers_matrices_lst

    def __back_propagation (self, Y: np.ndarray, layers_matrices_lst: list[tuple[np.ndarray, np.ndarray]]):
        delta = [(layers_matrices_lst[-1][1] - Y) / Y.shape[0]]
        grad_weights = [layers_matrices_lst[-2][1].T @ delta[-1]]

        delta.append(delta[-1] @ self.__output_weights.T * self.__leaky_relu_grad(layers_matrices_lst[-2][0]))
        grad_weights.append(layers_matrices_lst[-3][1].T @ delta[-1])

        for i in range(self.__layers - 2, -1, -1):
            delta.append(delta[-1] @ self.__hidden_weights[i + 1].T * self.__leaky_relu_grad(layers_matrices_lst[i + 1][0]))
            grad_weights.append(layers_matrices_lst[i][1].T @ delta[-1])

        delta.reverse()
        grad_weights.reverse()

        self.__output_weights -= self.__learning_rate * grad_weights[-1]
        self.__output_bias -= self.__learning_rate * np.sum(delta[-1], axis=0, keepdims=True)

        for i in range(self.__layers):
            self.__hidden_weights[i] -= self.__learning_rate * grad_weights[i]
            self.__hidden_bias[i] -= self.__learning_rate * np.sum(delta[i], axis=0, keepdims=True)

    def loss_function (self, Y: np.ndarray, predict_matrix: np.ndarray):
        predict_matrix = np.clip(predict_matrix, 1e-15, 1.0 - 1e-15)
        return - 1 / predict_matrix.shape[0] * np.sum(np.sum(Y * np.log(predict_matrix), axis=1, keepdims=True), axis=0)

    def fit (self, X: np.ndarray, Y: np.ndarray, epoch: int):
        last_error_value = 1e10
        self.__weight_init(X, Y)
        
        for i in range(epoch):
            layers_matrices_lst = self.__forward_propagation(X)
            error_value = self.loss_function(Y, layers_matrices_lst[-1][1])

            if i > 0 and np.abs(last_error_value - error_value) < self.__epsilon:
                break

            last_error_value = error_value
            self.__back_propagation(Y, layers_matrices_lst)