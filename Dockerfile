FROM tensorflow/tensorflow:1.0.0-py3
RUN apt-get -y update
RUN apt-get -y install libhdf5-dev
RUN apt-get -y install python3-setuptools
RUN easy_install3 pip
RUN apt-get -y install zlib1g-dev
COPY . /usr/local
WORKDIR /usr/local
RUN pip3 install textract --no-dependencies
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["main.py"]

