FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD exec uvicorn app:app --host 0.0.0.0 --port $PORT
