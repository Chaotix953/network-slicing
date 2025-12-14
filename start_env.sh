#!/bin/bash

# Nom du réseau Docker (à adapter si besoin, vérifiez avec 'docker network ls')
NET_NAME="network-slicing-second_sdn"

echo "=== 1. Nettoyage des anciens conteneurs ==="
# On force la suppression de mininet s'il existe déjà pour éviter le conflit de nom
sudo docker rm -f mininet 2>/dev/null || true

echo "=== 2. Démarrage du Contrôleur Ryu ==="
# On s'assure que Ryu est bien lancé via le compose
sudo docker compose up -d ryu

# Petite pause pour laisser le temps à Ryu de s'initialiser
echo "⏳ Attente du démarrage de Ryu (5s)..."
sleep 5

echo "=== 3. Démarrage de Mininet ==="
# On lance Mininet en mode DETACHÉ (-d) pour qu'il tourne en fond
# On lance directement la topologie (mn ...) au lieu de bash
sudo docker run -d --privileged \
  --name mininet \
  --network $NET_NAME \
  -v /lib/modules:/lib/modules:ro \
  iwaseyusuke/mininet:latest \
  mn --controller=remote,ip=ryu,port=6633 --switch=ovsk,protocols=OpenFlow13 --topo=single,3

echo "⏳ Attente de l'initialisation de la topologie (5s)..."
sleep 5

echo "=== 4. Configuration de la QoS (Queues) ==="
# On copie et exécute le script de configuration des queues DANS le conteneur
sudo docker cp scripts/setup_qos_queues.sh mininet:/setup_qos_queues.sh
sudo docker exec mininet chmod +x /setup_qos_queues.sh
sudo docker exec mininet /setup_qos_queues.sh s1

echo ""
echo "✅ Environnement prêt !"
echo "---------------------------------------------------"
echo "Pour voir les logs Mininet : sudo docker logs -f mininet"
echo "Pour entrer dans Mininet   : sudo docker exec -it mininet mn"
echo "Pour arrêter tout          : sudo docker rm -f mininet && sudo docker compose stop"
echo "---------------------------------------------------"