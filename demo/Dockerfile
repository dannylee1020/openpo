FROM python:3.12-slim

RUN apt-get update && apt-get install -y

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1

COPY pyproject.toml poetry.lock ./

RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root

RUN mkdir /app/peony

COPY ./peony /app/peony
COPY ./demo/ui.py .
EXPOSE 8501

CMD ["streamlit", "run", "ui.py"]