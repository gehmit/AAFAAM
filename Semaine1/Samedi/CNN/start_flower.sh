#!/bin/bash

# Activer l'environnement virtuel
source ../1ereSamedi/.venv/bin/activate

# Lancer le serveur en arrière-plan
echo "Démarrage du serveur..."
python server.py > server.log 2>&1 &

# Attendre quelques secondes que le serveur démarre
sleep 5

# Lancer les 4 clients en arrière-plan
for i in {0..3}
do
  echo "Démarrage du client $i..."
  python client.py $i > client_$i.log 2>&1 &
done

echo "Tous les clients ont été lancés."
