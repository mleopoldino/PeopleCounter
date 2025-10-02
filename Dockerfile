# PeopleCounter - Dockerfile (CPU base)
#
# Uso típico:
#   docker build -t people-counter:cpu .
#   docker run --rm -v $(pwd)/data:/app/data people-counter:cpu \
#       python src/app.py --source data/samples/demo.mp4

FROM python:3.10-slim

# Instala dependências do sistema necessárias para OpenCV/Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia apenas arquivos essenciais para instalar dependências primeiro
COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia o restante do código
COPY . .

# Diretórios de saída recomendados
RUN mkdir -p data/samples logs

# Comando padrão: mostra ajuda para lembrar opções disponíveis
CMD ["python", "src/app.py", "--help"]
