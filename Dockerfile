FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y libsndfile1
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]

