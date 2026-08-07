# Use Python 3.12 as the base image
FROM python:3.12-slim

# Create and switch to the /app folder inside the container
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install all required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the remaining project files
COPY . .

# Tell Docker that the application uses port 8000
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
