FROM python:3.10-slim

# Используем российский mirror и IPv4 для apt
RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4 \
    && sed -i 's|http://deb.debian.org/debian|http://mirror.yandex.ru/debian|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y build-essential python3-dev libopenblas-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./src ./src

CMD ["python", "-m", "src.main"] 