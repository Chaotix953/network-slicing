FROM python:3.9-slim

# Mise à jour de base
RUN apt update && apt upgrade -y

# Dépendances nécessaires pour compiler / installer certains paquets Python
RUN apt install -y --no-install-recommends \
        gcc \
        python3-dev \
        libffi-dev \
        libssl-dev \
        make \
        git \
    && rm -rf /var/lib/apt/lists/*

# Met à jour pip
RUN pip install --upgrade pip

# Fixer setuptools / wheel à des versions raisonnables
RUN pip install "setuptools<60" "wheel<1.0"

# Récupérer une version récente de Ryu compatible Python 3
WORKDIR /tmp
RUN git clone https://github.com/faucetsdn/ryu.git

# Installer Ryu depuis les sources (Python 3)
WORKDIR /tmp/ryu
RUN pip install --no-build-isolation .

# (Optionnel) Installer oslo.config si besoin
RUN pip install oslo.config

# Répertoire de travail pour nos apps
WORKDIR /apps

# Copier notre application Ryu (simple_switch_13.py, etc.)
COPY apps/ /apps

EXPOSE 6653

# Lancer Ryu via le module Python
CMD ["python", "-m", "ryu.cmd.manager", "simple_switch_13.py"]
