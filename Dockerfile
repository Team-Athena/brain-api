FROM tensorflow/tensorflow

RUN apt update -y
RUN apt install python3-pip -y
RUN pip install flask \
                flask_cors \
                numpy \
                pandas \
                matplotlib \
                nilearn \ 
                pickle-mixin \ 
                boto3

WORKDIR /app

EXPOSE 5000
