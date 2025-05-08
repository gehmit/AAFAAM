
# Federated Learning (Flower)

Ce projet montre un exemple simple d'entraînement fédéré d'un modèle CNN à l'aide de la bibliothèque [Flower](https://flower.dev/).

## Prérequis

- Python 3.x+
- pip
- bash

## Installation

Installez les dépendances avec :

```bash
pip install -r requirements.txt
````

## Lancement du projet

1. Donnez les droits d'exécution au script de démarrage :

```bash
chmod +x start_flower.sh
```

2. Lancez le script :

```bash
./start_flower.sh
```

Ce script démarre :

* Le serveur fédéré (`server.py`)
* Deux clients (`client.py`) simulés

## Structure du projet

* `server.py` : serveur central Flower
* `client.py` : client entraînant un CNN localement
* `requirements.txt` : dépendances Python
* `start_flower.sh` : script de démarrage



