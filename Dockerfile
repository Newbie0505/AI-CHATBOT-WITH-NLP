# Standard python base image
FROM python:3.10-slim

# Install system-level audio development libraries for SpeechRecognition
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Installs all your original requirements perfectly
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]