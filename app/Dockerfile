FROM debian:latest

# Working directory for the application
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    lsb-release \

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_19.x | bash -
RUN apt-get install -y nodejs npm

# Confirm Node.js and npm installation
RUN node -v && npm -v

# Install Python
RUN apt-get install -y python3.11 python3-pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Confirm Python installation
RUN python --version && pip --version

# Install Python dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --break-system-packages -r requirements.txt

# Installing node dependencies
COPY ./frontend/my-app2/package.json /app/frontend/my-app2/package.json
WORKDIR /app/frontend/my-app2

RUN npm install

# Copy app
COPY . /app

RUN apt-get install -y supervisor

# Copying supervisor file
COPY ./supervisord.conf /etc/supervisor/conf.d/supervisord.conf

WORKDIR /app
