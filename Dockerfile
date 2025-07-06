# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all project files to the container
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI's default port
EXPOSE 8000

# Start the app using Uvicorn
CMD ["uvicorn", "Hardcodewallmart_rag:app", "--host", "0.0.0.0", "--port", "8000"]
