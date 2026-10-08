import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.multilayer_perceptron.layers import DenseLayer
from src.multilayer_perceptron.network import Network


TRAIN_PATH = "data/train.csv"
VALID_PATH = "data/valid.csv"
MODEL_PATH = "data/saved_model.npz"
 
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.01
SEED = 42
 
train = pd.read_csv(TRAIN_PATH)
valid = pd.read_csv(VALID_PATH)
 
X_train = train.drop(columns=["id", "diagnosis"]).to_numpy(dtype=float)
X_valid = valid.drop(columns=["id", "diagnosis"]).to_numpy(dtype=float)
 
y_train = (train["diagnosis"] == "M").astype(int).to_numpy()
y_valid = (valid["diagnosis"] == "M").astype(int).to_numpy()
Y_train, Y_valid = np.eye(2)[y_train], np.eye(2)[y_valid]
 
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mean) / std
X_valid = (X_valid - mean) / std
 
print("x_train shape :", X_train.shape)
print("x_valid shape :", X_valid.shape)
 
rng = np.random.default_rng(SEED)
 
layers = [DenseLayer(30, 16, "relu", "kaiming_normal", rng),
          DenseLayer(16, 8, "relu", "kaiming_normal", rng),
          DenseLayer(8, 2, "softmax", "xavier_normal", rng)]
 
net = Network(layers, "categorical_cross_entropy", "sgd", LEARNING_RATE, rng)
history = net.fit(X_train, Y_train, EPOCHS, BATCH_SIZE, validation=(X_valid, Y_valid))
net.save(MODEL_PATH, mean=mean, std=std)
 
epochs = range(1, EPOCHS + 1)
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
 
ax_loss.plot(epochs, history["loss"], label="training loss")
ax_loss.plot(epochs, history["val_loss"], "--", label="validation loss")
ax_loss.set_xlabel("epochs")
ax_loss.set_ylabel("loss")
ax_loss.legend()
ax_loss.grid(True)
 
ax_acc.plot(epochs, history["acc"], label="training acc")
ax_acc.plot(epochs, history["val_acc"], label="validation acc")
ax_acc.set_xlabel("epochs")
ax_acc.set_ylabel("accuracy")
ax_acc.legend()
ax_acc.grid(True)
 
fig.suptitle("Learning Curves")
plt.show()