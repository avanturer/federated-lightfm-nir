FROM mambaorg/micromamba:1.5.7

ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Ставим системные зависимости (Java для PyFlink — опционально)
USER root
RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4 \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Ставим Python и библиотеки через conda-forge, включая LightFM (без сборки из исходников)
RUN micromamba install -y -n base -c conda-forge \
      python=3.10 \
      lightfm=1.17 \
      numpy=1.24 \
      scipy=1.10 \
      pandas \
      scikit-learn \
      matplotlib \
      openblas \
    && micromamba clean -a -y

# Рабочая директория
WORKDIR /app

# Устанавливаем оставшиеся Python-зависимости через pip (CPU-версия PyTorch)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
         flwr==1.8.0 \
         torch==2.2.2

ENV OUTPUT_DIR=/app/out

# Копируем код и entrypoint
COPY ./src ./src
COPY ./federated_flower ./federated_flower
COPY ./docker_entry.sh /app/docker_entry.sh
RUN chmod +x /app/docker_entry.sh

# Запускаем под root, чтобы гарантировать права записи в смонтированный volume
USER root

# Точка входа — подготавливает OUTPUT_DIR и запускает main
ENTRYPOINT ["/bin/bash", "/app/docker_entry.sh"]