import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
import numpy as np

class MnistClient(fl.client.NumPyClient):
    def __init__(self):
        # Charger les données Iris
        data = load_breast_cancer()
        X = data.data
        y = data.target
        # On ne garde que deux classes pour la classification binaire (0 et 1)
        mask = y < 2
        self.X_train = X[mask]
        self.y_train = y[mask]

        # Standardisation (important pour LogisticRegression)
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)

        self.model = LogisticRegression(max_iter=200)
        self.model.fit(self.X_train, self.y_train)

    def get_parameters(self, config=None):
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        y_pred = self.model.predict_proba(self.X_train)
        loss = log_loss(self.y_train, y_pred)
        accuracy = self.model.score(self.X_train, self.y_train)
        return loss, len(self.X_train), {"accuracy": accuracy}


fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=MnistClient())
