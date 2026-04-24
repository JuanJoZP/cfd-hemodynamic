FROM dolfinx/dolfinx:v0.9.0

LABEL Author="cfd-hemodynamic"
LABEL Version="0.1.0"

ENV DEBIAN_FRONTEND=noninteractive

# dependencias uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    tar \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /opt/project
COPY pyproject.toml .

# ignorando lo instalado para evitar conflictos con dolfinx
RUN uv export --format requirements-txt --output-file requirements.txt && \
    pip install --ignore-installed --no-cache-dir -r requirements.txt

# fenicsx-pctools
WORKDIR /opt/deps
RUN wget https://gitlab.com/rafinex-external-rifle/fenicsx-pctools/-/archive/v0.9.1/fenicsx-pctools-v0.9.1.tar && \
    tar -xf fenicsx-pctools-v0.9.1.tar && \
    cd fenicsx-pctools-v0.9.1 && \
    python3 -m pip install .

WORKDIR /opt/project

CMD ["/bin/bash"]
