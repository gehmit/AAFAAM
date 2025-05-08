import flwr as fl
import tensorflow as tf
from typing import List
import numpy as np

def get_eval_fn():
    # Charger les données de test globales (valeurs médianes fictives ici)
    import medmnist
    from medmnist import PathMNIST
    from medmnist import INFO
    from tensorflow.keras.utils import to_categorical

    info = INFO["pathmnist"]
    DataClass = PathMNIST

    test_data = DataClass(split='test', download=True)
    x_test = tf.convert_to_tensor(test_data.imgs, dtype=tf.float32) / 255.0
    y_test = to_categorical(test_data.labels, num_classes=9)

    def evaluate(server_round, parameters, config):
        model = build_model()
        model.set_weights(parameters)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        return loss, {"accuracy": acc}

    return evaluate

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(9, activation="softmax"),
    ])
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    return model

strategy = fl.server.strategy.FedAvg(
    evaluate_fn=get_eval_fn(),
    min_available_clients=2,
)

# Lancement du serveur avec les bons arguments nommés
fl.server.start_server(
    server_address="127.0.0.1:8085",
    strategy=strategy,  
    config=fl.server.ServerConfig(num_rounds=20)
)
