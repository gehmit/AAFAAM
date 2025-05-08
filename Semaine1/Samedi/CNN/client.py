import flwr as fl
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from medmnist import PathMNIST
from medmnist import INFO

class MedClient(fl.client.NumPyClient):
    def __init__(self):
        info = INFO["pathmnist"]
        DataClass = PathMNIST

        train_data = DataClass(split='train', download=True)
        self.x_train = tf.convert_to_tensor(train_data.imgs, dtype=tf.float32) / 255.0
        self.y_train = to_categorical(train_data.labels, num_classes=9)

        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(9, activation="softmax"),
        ])
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        return model

    def get_parameters(self, config=None):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=32, verbose=0)
        return self.get_parameters(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        return loss, len(self.x_train), {"accuracy": accuracy}

# Appel corrigé de start_numpy_client avec des arguments nommés
fl.client.start_numpy_client(
    server_address="127.0.0.1:8085",
    client=MedClient()
)
