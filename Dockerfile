FROM ubuntu:20.04

RUN apt-get update \
    && apt-get install -y python3.8\
    && apt-get install -y pip
RUN pip install --upgrade pip

COPY . .

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

EXPOSE ${MY_SERVICE_PORT}:5000

ENTRYPOINT ["python3"]

CMD ["app.py"]