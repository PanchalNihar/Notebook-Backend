# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
