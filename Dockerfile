# Use an official Python runtime as a parent image
# We choose 'slim' because it's a smaller, more lightweight version
FROM python:3.11-slim

# Set the working directory inside the container to /app
# This is where our code will live
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching.
# This step will only be re-run if requirements.txt changes.
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code (main.py, app.py, etc.)
# into the container's /app directory
COPY . .

# Expose port 8501. This tells Docker that the container will listen on this port.
EXPOSE 8501

# Define the command to run the application when the container starts.
# We run main.py because it contains the Streamlit UI.
CMD ["streamlit", "run", "main.py"]

# --- Terminal Commands ---
#
# 1. Build the Docker image (run this command from your project folder):
#    docker build -t my-rag-app .
#
# 2. Run the container from the image you just built:
#    docker run -p 8501:8501 my-rag-app
#
# After running the second command, open your web browser and go to:
# http://localhost:8501
