#!/bin/bash

# Install dependencies for building CmdStan
sudo apt-get update && sudo apt-get install -y build-essential

# Set the version of CmdStan you wish to install
CMDSTAN_VERSION="2.35.0"

# Download and extract CmdStan
wget https://github.com/stan-dev/cmdstan/releases/download/v${CMDSTAN_VERSION}/cmdstan-${CMDSTAN_VERSION}.tar.gz
tar -xzf cmdstan-${CMDSTAN_VERSION}.tar.gz

# Build CmdStan
cd cmdstan-${CMDSTAN_VERSION}
make build -j4  # Adjust '4' based on available CPU cores
cd ..

# Set CMDSTAN environment variable
export CMDSTAN="C:\Users\danie\.cmdstan-${CMDSTAN_VERSION}"
echo "CmdStan installed at $CMDSTAN"
