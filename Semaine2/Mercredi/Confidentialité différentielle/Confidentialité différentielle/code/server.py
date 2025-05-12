import flwr as fl
from typing import List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

# Charger le dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Filtrer les classes 0 et 1
mask = y < 2
X = X[mask]
y = y[mask]

# Standardiser
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialiser le modèle
model = LogisticRegression(max_iter=200)
model.fit(X, y)

def get_eval_fn(model):
    def evaluate(server_round: int, parameters: List[np.ndarray], config: dict):
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]
        y_pred = model.predict_proba(X)
        return log_loss(y, y_pred), {"accuracy": model.score(X, y)}
    return evaluate

strategy = fl.server.strategy.FedAvg(
    min_available_clients=2,
    evaluate_fn=get_eval_fn(model),
)

fl.server.start_server(
    server_address="127.0.0.1:8080",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=10)
)
