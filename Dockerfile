FROM python:3.10-slim

WORKDIR /app

# Stáhneš requirements
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

# Nakopíruješ zbytek
COPY . .

# Exponuješ port pro API
EXPOSE 7860

# Spouštíš appku – uprav podle svého backendu
CMD ["python", "main.py"]
