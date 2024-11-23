# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Install build tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CmdStan
ENV CMDSTAN_VERSION=2.35.0
RUN wget https://github.com/stan-dev/cmdstan/releases/download/v${CMDSTAN_VERSION}/cmdstan-${CMDSTAN_VERSION}.tar.gz \
    && tar -xzf cmdstan-${CMDSTAN_VERSION}.tar.gz \
    && rm cmdstan-${CMDSTAN_VERSION}.tar.gz \
    && cd cmdstan-${CMDSTAN_VERSION} \
    && make build -j4 \
    && cd /app

# Set CMDSTAN environment variable
ENV CMDSTAN=/app/cmdstan-${CMDSTAN_VERSION}


# Set the working directory to /app/src
WORKDIR /app/src

# Copy your application code into the container
COPY . /app/src

RUN chmod +x . /app/src/bounded_back
RUN chmod +x . /app/src/moving_se_sp

# Expose the port your Dash app will run on
EXPOSE 8050


# Command to run your application
CMD ["python", "app.py"]
