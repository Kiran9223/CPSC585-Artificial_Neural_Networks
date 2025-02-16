# Use an official CUDA-enabled Python base image
FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Copy your CuPy code into the container
COPY . .

# Define the command to run your application
CMD ["python3", "Regression_withGPU_Numba.py"]
