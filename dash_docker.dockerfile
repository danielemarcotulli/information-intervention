# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Install build tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    make \
    g++ \
    libtbb2 \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CmdStan
ENV CMDSTAN_VERSION=2.35.0
RUN mkdir -p /opt && \
    cd /opt && \
    wget https://github.com/stan-dev/cmdstan/releases/download/v${CMDSTAN_VERSION}/cmdstan-${CMDSTAN_VERSION}.tar.gz && \
    tar -xzvf cmdstan-${CMDSTAN_VERSION}.tar.gz && \
    rm cmdstan-${CMDSTAN_VERSION}.tar.gz && \
    mv cmdstan-${CMDSTAN_VERSION} cmdstan && \
    cd cmdstan && \
    make build -j4

# Set environment variable for CmdStan path
ENV CMDSTAN=/opt/cmdstan

# Copy your application code into the container
COPY . /app

# Set the working directory to /app/src
WORKDIR /app/src

# Compile Stan models
RUN /opt/cmdstan/bin/stanc --version && \
    make bounded_back moving_se_sp

# Ensure the Stan model executables are executable
RUN chmod +x /app/src/bounded_back
RUN chmod +x /app/src/moving_se_sp

RUN chmod 755 /app/src/bounded_back /app/src/moving_se_sp

RUN mv /app/src/bounded_back /usr/local/bin/ && \
    mv /app/src/moving_se_sp /usr/local/bin/

# Expose the port your Dash app will run on
EXPOSE 8050

# Command to run your application
CMD ["python", "app.py"]
