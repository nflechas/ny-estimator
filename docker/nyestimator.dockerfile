# Use the official Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables to avoid Python buffering and minimize output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set up a working directory
RUN mkdir -p /opt/project
WORKDIR /opt/project

# Install system dependencies (optional, based on your project needs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files to the container
COPY . /opt/project

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt