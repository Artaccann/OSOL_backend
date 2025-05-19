FROM python:3.10-slim

# Pracovní složka
WORKDIR /app

# Zkopíruj requirements a nainstaluj závislosti
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

# Zkopíruj zbytek projektu
COPY . .

# Port pro FastAPI
EXPOSE 7860

# Start přes uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
