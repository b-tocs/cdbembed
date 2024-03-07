FROM python:3.10-slim-bookworm

WORKDIR /app

COPY requirements.txt .
COPY src/*.py .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000

ENV DEFAULT_MODEL all-MiniLM-L6-v2
ENV OLLAMA_URL http://localhost:8000
ENV OLLAMA_MODEL llama2


CMD ["python", "/app/main.py"]