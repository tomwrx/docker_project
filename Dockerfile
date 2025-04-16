#  Official Python base image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy all source code 
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port Flask runs on
EXPOSE 5000

# Start the Flask app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "flask_app.app:app"]