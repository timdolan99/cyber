# 1. Base image matching Python 3.12
FROM python:3.12-slim

# 2. Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 3. Set working directory
WORKDIR /app

# 4. Install system dependencies required for C++ builds (ChromaDB, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy application source code
COPY . .

# 7. Expose Streamlit default port
EXPOSE 8501

# 8. Configure health check for Azure Container Apps / Web Apps
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 9. Launch Streamlit app
CMD ["streamlit", "run", "teacher_app_v2.py", "--server.port=8501", "--server.address=0.0.0.0"]