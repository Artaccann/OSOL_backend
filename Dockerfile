FROM python:3.10-slim

# Nastavení pracovní složky
WORKDIR /app

# Zkopíruj celý projekt (včetně handler.py a .runpod.yml)
COPY . .

# Aktualizuj pip a nainstaluj závislosti
RUN pip install --upgrade pip && \
    pip install torch==2.1.2+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install transformers peft unsloth fastapi uvicorn python-dotenv

# Volitelně: zkontroluj model existenci
# RUN echo "MODEL:" $HF_MODEL_NAME

# Endpoint očekává handler.handler → neuvádíš žádný CMD!
CMD ["handler.handler"]

