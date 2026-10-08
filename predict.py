import numpy as np
import pandas as pd

from src.multilayer_perceptron.network import Network
from src.multilayer_perceptron.loss_functions import get_loss_function


MODEL_PATH = "data/saved_model.npz"
DATA_PATH = "data/valid.csv"

net, extra = Network.load(MODEL_PATH)
mean, std = extra["mean"], extra["std"]

data = pd.read_csv(DATA_PATH)

X = data.drop(columns=["id", "diagnosis"]).to_numpy(dtype=float)
X = (X - mean) / std

y = (data["diagnosis"] == "M").astype(int).to_numpy()
Y = np.eye(2)[y]

print("x shape :", X.shape)

prediction = net.predict(X)

loss = get_loss_function("binary_cross_entropy").forward(prediction, Y)
accuracy = (prediction.argmax(axis=1) == y).mean()

print(f"binary cross-entropy : {loss:.4f}")
print(f"accuracy : {accuracy:.4f}")
