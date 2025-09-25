FROM python:3.13-slim

WORKDIR agent

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN SECRET=$(chainlit create-secret | tail -n 1)
RUN echo "CHAINLIT_SECRET=$SECRET" > .env

RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://ollama.com/install.sh | sh

RUN ollama serve & \
    sleep 5 && \
    ollama pull llama3.2:3b-instruct-fp16 && \
    ollama create rag_agent -f Modelfile

CMD ["chainlit", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
