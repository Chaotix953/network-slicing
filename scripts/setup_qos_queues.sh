#!/bin/bash
#
# Script de configuration des queues QoS dans Open vSwitch
# Pour le slicing 5G (URLLC + mMTC)
#
# Usage: ./setup_qos_queues.sh [switch_name]
#

set -e  # Exit on error

# Configuration
SWITCH_NAME="${1:-s1}"
MAX_RATE_MBPS=100  # Bande passante totale du lien

# Conversion Mbps -> bps
MAX_RATE=$((MAX_RATE_MBPS * 1000000))

# Configuration des queues (en bps)
URLLC_MIN_RATE=$((10 * 1000000))   # 10 Mbps minimum
URLLC_MAX_RATE=$((50 * 1000000))   # 50 Mbps maximum

mMTC_MIN_RATE=$((1 * 1000000))     # 1 Mbps minimum
mMTC_MAX_RATE=$((50 * 1000000))    # 50 Mbps maximum

echo "=============================================="
echo "Configuration QoS pour Network Slicing 5G"
echo "=============================================="
echo "Switch: $SWITCH_NAME"
echo "Bande passante totale: ${MAX_RATE_MBPS} Mbps"
echo ""

# Vérifier que le switch existe
if ! ovs-vsctl list-br | grep -q "^${SWITCH_NAME}$"; then
    echo "ERREUR: Le switch '$SWITCH_NAME' n'existe pas"
    echo "Switches disponibles:"
    ovs-vsctl list-br
    exit 1
fi

# Lister les ports du switch
echo "Ports du switch $SWITCH_NAME:"
PORTS=$(ovs-vsctl list-ports "$SWITCH_NAME")
echo "$PORTS"
echo ""

# Pour chaque port, créer les queues QoS
for PORT in $PORTS; do
    echo "----------------------------------------"
    echo "Configuration du port: $PORT"
    echo "----------------------------------------"
    
    # Supprimer l'ancienne config QoS si elle existe
    echo "1. Nettoyage de l'ancienne configuration..."
    ovs-vsctl -- --if-exists clear Port "$PORT" qos 2>/dev/null || true
    
    # Supprimer les anciennes queues
    QUEUE_IDS=$(ovs-vsctl --columns=_uuid --bare find Queue | tr '\n' ' ')
    for QUEUE_ID in $QUEUE_IDS; do
        ovs-vsctl destroy Queue "$QUEUE_ID" 2>/dev/null || true
    done
    
    QOS_IDS=$(ovs-vsctl --columns=_uuid --bare find QoS | tr '\n' ' ')
    for QOS_ID in $QOS_IDS; do
        ovs-vsctl destroy QoS "$QOS_ID" 2>/dev/null || true
    done
    
    echo "2. Création des queues QoS..."
    
    # Créer les queues avec HTB (Hierarchical Token Bucket)
    ovs-vsctl -- set Port "$PORT" qos=@newqos \
        -- --id=@newqos create QoS type=linux-htb \
           other-config:max-rate="$MAX_RATE" \
           queues:1=@q_urllc \
           queues:2=@q_mmtc \
        -- --id=@q_urllc create Queue \
           other-config:min-rate="$URLLC_MIN_RATE" \
           other-config:max-rate="$URLLC_MAX_RATE" \
        -- --id=@q_mmtc create Queue \
           other-config:min-rate="$mMTC_MIN_RATE" \
           other-config:max-rate="$mMTC_MAX_RATE"
    
    echo "   ✓ Queue 1 (URLLC): min=${URLLC_MIN_RATE}bps, max=${URLLC_MAX_RATE}bps"
    echo "   ✓ Queue 2 (mMTC):  min=${mMTC_MIN_RATE}bps, max=${mMTC_MAX_RATE}bps"
    echo ""
done

echo "=============================================="
echo "Configuration terminée avec succès !"
echo "=============================================="
echo ""

# Afficher la configuration finale
echo "Vérification de la configuration:"
echo ""

echo "1. QoS configurés:"
ovs-vsctl list qos

echo ""
echo "2. Queues configurées:"
ovs-vsctl list queue

echo ""
echo "=============================================="
echo "Pour vérifier les statistiques des queues:"
echo "  ovs-ofctl -O OpenFlow13 queue-stats $SWITCH_NAME"
echo ""
echo "Pour voir les flows avec queues assignées:"
echo "  ovs-ofctl -O OpenFlow13 dump-flows $SWITCH_NAME"
echo "=============================================="
