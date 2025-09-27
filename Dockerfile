FROM base_docker:latest

WORKDIR agent

COPY . .

RUN SECRET=$(chainlit create-secret | tail -n 1)
RUN echo "CHAINLIT_SECRET=$SECRET" > .env

CMD ["chainlit", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
