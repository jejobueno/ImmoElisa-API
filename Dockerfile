FROM ubuntu:20.04

RUN apt-get update \
    && apt-get install -y python3.8\
    && apt-get install -y pip
RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

CMD ["python", "code/hello_world.py"]