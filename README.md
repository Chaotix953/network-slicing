# 5G Network Slicing — Allocation dynamique URLLC / mMTC (Deep RL)

Ce dépôt implémente un système de simulation et de contrôle pour l'allocation dynamique de bande passante entre deux slices (URLLC et mMTC) à l'aide d'agents d'apprentissage par renforcement (PPO, SAC, TD3).

- Langage : Python 3.10+
- Principales dépendances : `gymnasium`, `stable-baselines3`, `torch`, `numpy`, `pyyaml`, `matplotlib`, `mininet`, `ryu`

**Structure**
- `configs/` : fichiers YAML de configuration (réseau, trafic, entraînement)
- `src/` : code source (envs, agents, training, infrastructure, traffic, utils)
- `scripts/` : scripts d'utilité (ex. `train.py`, configuration OVS)
- `ryu/apps/` : contrôleur Ryu (`slice_switch_13.py`)
- `docker-compose.yml` : environnement Ryu + (optionnel) autres services
- `data/`, `results_raw/`, `experiments/` : résultats et checkpoints

## Installation rapide

```bash
# Cloner
git clone <votre-repo-url>
cd network-slicing

# Créer et activer un environnement Python (recommandé)
python -m venv venv
source venv/bin/activate

# Installer dépendances
pip install -r requirements.txt
```

Remarque : certains composants (Mininet, Ryu) sont généralement exécutés en conteneurs Docker (voir la section Docker).

## Exemples d'utilisation

## Scripts utiles

Un script pratique est fourni dans `scripts/start_env.sh` pour initialiser rapidement l'environnement de développement :

```bash
# Créer/activer venv, installer dépendances, démarrer docker-compose et copier le contrôleur Ryu
./scripts/start_env.sh

# Options utiles :
#  --no-docker    : ne lance pas Docker Compose
#  --no-ryu-copy  : n'essaie pas de copier ryu/apps/slice_switch_13.py dans le conteneur 'ryu'
#  --no-venv      : ne crée/active pas l'environnement virtuel ni n'installe les dépendances
```

Ce script est idempotent et utile pour préparer rapidement une machine de développement.


### Entraînement en mode simulation (pas de Ryu requis)

```bash
# Entraîner avec PPO (100k timesteps)
python scripts/train.py --algorithm ppo --timesteps 100000 --simulation

# Entraîner SAC et comparer aux baselines (simulation)
python scripts/train.py --algorithm sac --timesteps 50000 --compare-baselines --simulation
```

### Évaluation d'un modèle entraîné

```bash
python scripts/train.py --eval-only --load-model path/to/model.zip --simulation
```

### Entraîner avec le réseau réel (Ryu)

1. Démarrer les conteneurs Docker :

```bash
sudo docker compose up -d
```

2. Copier le contrôleur Ryu corrigé dans le conteneur (TRÈS IMPORTANT : faire avant de redémarrer Ryu)

```bash
sudo docker cp ryu/apps/slice_switch_13.py ryu:/apps/slice_switch_13.py
sudo docker restart ryu
```

3. Lancer Mininet (dans un conteneur) :

```bash
sudo docker run --rm -it --privileged \
  --network 5g-slicing-drl_sdn \
  -v /lib/modules:/lib/modules:ro \
  --name mininet \
  iwaseyusuke/mininet:latest bash

# puis dans le conteneur Mininet
mn --controller=remote,ip=ryu,port=6633 --switch=ovsk,protocols=OpenFlow13 --topo=single,3
```

4. Configurer les queues OVS :

```bash
sudo docker cp scripts/setup_qos_queues.sh mininet:/setup_qos_queues.sh
sudo docker exec -it mininet bash -lc "/setup_qos_queues.sh s1"
```

5. Lancer l'entraînement en pointant vers l'API Ryu :

```bash
python scripts/train.py --algorithm ppo --timesteps 100000 --ryu-url http://localhost:8080
```

## Configuration

- Fichiers principaux de configuration : `configs/default.yaml`, `configs/network.yml`, `configs/traffic_config.yml`.
- Les options CLI (dans `scripts/train.py`) permettent de surcharger certains paramètres (algorithm, timesteps, ryu-url, config, etc.).

## Tests

Le projet contient quelques tests unitaires (`tests/`). Pour exécuter la suite de tests :

```bash
pytest
```

Note : certains tests d'intégration peuvent nécessiter Docker/Mininet/Ryu et peuvent échouer si ces services ne sont pas disponibles.

## Bonnes pratiques et prochains pas

- Toujours copier `ryu/apps/slice_switch_13.py` dans le conteneur Ryu avant de démarrer/ redémarrer le service (guide détaillé dans `GUIDE_DEMARRAGE.md`).
- Personnaliser `configs/default.yaml` pour expérimentations reproductibles.
- Sauvegarder/archiver les checkpoints dans `data/checkpoints/` et les résultats dans `results_raw/`.

## Contribution

- Fork → branch → PR. Merci d'ajouter des tests pour les nouvelles fonctionnalités.
- Ouvrir des issues pour bugs ou demandes d'amélioration.

## Comment push sur GitHub (rapide)

```bash
# Initialiser git (si nécessaire)
git init
git add .
git commit -m "Initial commit"
# Ajouter l'origine distante
git remote add origin git@github.com:utilisateur/repo.git
git branch -M main
git push -u origin main
```


## Licence

Ce dépôt est publié sous licence MIT. Voir le fichier `LICENSE` à la racine du projet. Pensez à mettre à jour le copyright dans `LICENSE`.

