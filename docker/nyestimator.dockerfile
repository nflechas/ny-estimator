FROM python:3.11-slim

# Set environment variables to avoid Python buffering and minimize output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set up a working directory
WORKDIR /opt/project

COPY requirements.txt /opt/project/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

CMD cd app && uvicorn --host 0.0.0.0 --port 8080 nyestimator_api:app --reload