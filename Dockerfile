# Use an official lightweight Python image
FROM python:3.10-slim

# Install system-level audio dependencies required for PyAudio compilation
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory inside the container
WORKDIR /app

# Copy your unchanged project files
COPY . /app

# Install dependencies from your updated requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Expose the standard port Streamlit uses
EXPOSE 8501

# Command to launch the app securely
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]