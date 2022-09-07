# syntax=docker/dockerfile:1
FROM python:3.10
WORKDIR /opt/syrchello
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY Model.py Model.py
COPY bot.py bot.py
