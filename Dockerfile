FROM python:3.11-slim

WORKDIR /complaint-project

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0+cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY deploy ./deploy
COPY src ./src
COPY src_bert ./src_bert
COPY src_embeddings ./src_embeddings
COPY models ./models

CMD ["uvicorn", "deploy.serve:app", "--host", "0.0.0.0", "--port", "8000"]
