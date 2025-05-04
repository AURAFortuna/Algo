# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY knn.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "knn.py"] 