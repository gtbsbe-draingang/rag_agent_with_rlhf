FROM python:3.13-slim

WORKDIR agent

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN SECRET=$(chainlit create-secret | tail -n 1)
RUN echo "CHAINLIT_SECRET=$SECRET" > .env

RUN ollama pull llama3.2:3b-instruct-fp16
RUN ollama create rag_agent -f Modelfile

CMD ["chainlit", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
