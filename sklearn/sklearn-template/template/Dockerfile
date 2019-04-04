FROM google/cloud-sdk:latest

RUN apt-get -y install pylint

WORKDIR /staging

ADD ./.pylintrc /staging/.pylintrc
ADD ./trainer /staging/trainer

RUN pylint ./trainer/*

