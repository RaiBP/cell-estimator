# base image
FROM nvidia/cuda:12.2.0-base-ubuntu20.04

# Working directory for the application
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

# Set the timezone
RUN echo "tzdata tzdata/Areas select Europe" | debconf-set-selections \
 && echo "tzdata tzdata/Zones/Europe select Berlin" | debconf-set-selections

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    lsb-release \
    tzdata \
    git \
    libgl1-mesa-glx 

# Install Node.js
RUN curl -sL https://deb.nodesource.com/setup_19.x | bash -
RUN apt-get install -y nodejs

# Confirm Node.js and npm installation
RUN node -v && npm -v

# Install software-properties-common to be able to add new repositories
RUN apt-get update && apt-get install -y software-properties-common

# Add the deadsnakes PPA
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y python3.10 python3.10-distutils python3.10-dev

# Set Python 3.10 as the default Python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Confirm Python installation
#RUN python --version && pip --version
RUN python3 --version && pip3 --version

# Install Python dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Installing node dependencies
COPY ./frontend/my-app2/package.json /app/frontend/my-app2/package.json
WORKDIR /app/frontend/my-app2
RUN npm install

# Copy app
COPY . /app

# Install supervisor
RUN apt-get install -y supervisor

# Copying supervisor file
COPY ./supervisord.conf /etc/supervisor/conf.d/supervisord.conf

WORKDIR /app
