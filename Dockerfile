FROM ubuntu

RUN apt update -y
RUN apt install python3-pip -y
RUN pip install flask flask_cors

WORKDIR /app

EXPOSE 5000

COPY . .
