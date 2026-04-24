# Basado en tu Bootstrap: docker / From: dolfinx/dolfinx:v0.9.0
FROM dolfinx/dolfinx:v0.9.0

# %labels
LABEL Author="cfd-hemodynamic"
LABEL Version="0.1.0"

# Evita interacciones durante la instalación de paquetes
ENV DEBIAN_FRONTEND=noninteractive

# %post - Parte 1: Instalación de dependencias de sistema y uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    tar \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instalamos uv globalmente
RUN pip install --no-cache-dir uv

# %files y %post - Parte 2: Manejo de dependencias del proyecto
WORKDIR /opt/project
COPY pyproject.toml .

# Exportamos e instalamos ignorando lo instalado para evitar conflictos con dolfinx
RUN uv export --format requirements-txt --output-file requirements.txt && \
    pip install --ignore-installed --no-cache-dir -r requirements.txt

# %post - Parte 3: Descarga e instalación de fenicsx-pctools
WORKDIR /opt/deps
RUN wget https://gitlab.com/rafinex-external-rifle/fenicsx-pctools/-/archive/v0.9.1/fenicsx-pctools-v0.9.1.tar && \
    tar -xf fenicsx-pctools-v0.9.1.tar && \
    cd fenicsx-pctools-v0.9.1 && \
    python3 -m pip install .

# Volvemos al directorio de trabajo del proyecto
WORKDIR /opt/project

# Comando por defecto al iniciar el contenedor
CMD ["/bin/bash"]
