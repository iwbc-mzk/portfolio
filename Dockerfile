FROM python:3.6

ENV PYTHONUNBUFFERED 1
ARG django_secret_key
ENV DJANGO_SECRET_KEY $django_secret_key

RUN mkdir /code
COPY . /code
WORKDIR /code

RUN pip install -r requirements.txt