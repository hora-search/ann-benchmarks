FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y python3-numpy python3-scipy python3-pip build-essential git
RUN pip3 install -U pip

WORKDIR /home/app
COPY requirements.txt run_algorithm.py ./
RUN pip3 install -r requirements.txt
RUN pip3 install numpy --upgrade
RUN pip3 install scipy --upgrade
RUN pip3 install scikit-learn --upgrade

ENTRYPOINT ["python3", "run_algorithm.py"]
