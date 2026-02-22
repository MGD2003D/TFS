FROM python:3.11

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt && \
    python -m spacy download ru_core_news_sm || \
    python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
