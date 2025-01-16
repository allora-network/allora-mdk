# Stage 1: Use the official allora-offchain-node image to extract the binary
FROM --platform=linux/amd64 alloranetwork/allora-offchain-node:v0.5.0 AS allora_node

# Stage 2: Use the official Python 3.9 slim image for the allora-worker
FROM python:3.9-slim

# Set environment variables to avoid installing CUDA dependencies
ENV TORCH_CUDA_ARCH_LIST=""
ENV CUDA_VISIBLE_DEVICES=""

# Create a non-privileged user
RUN addgroup --system appgroup && adduser --system --group appuser

# Install cron, curl, and supervisor
RUN apt-get update && apt-get install -y \
    cron \
    curl \
    supervisor \
    build-essential \
    gcc \
    g++ \
    libatlas-base-dev \
    python3-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default model, can be overridden during deployment
ENV MODEL=arima

# Set the working directory in the container
WORKDIR /app

# Copy the project into the working directory
COPY . /app/

# Upgrade pip and install Cython and numpy manually by parsing the requirements.txt
# they are known nuisances with the pip install -r requirements.txt
RUN pip install --upgrade pip && \
    SETUPTOOLS_VERSION=$(grep -i '^setuptools==' /app/requirements.txt | cut -d'=' -f3) \
    && CYTHON_VERSION=$(grep -i '^Cython==' /app/requirements.txt | cut -d'=' -f3) \
    && NUMPY_VERSION=$(grep -i '^numpy==' /app/requirements.txt | cut -d'=' -f3) \
    && PYSTAN_VERSION=$(grep -i '^pystan==' /app/requirements.txt | cut -d'=' -f3) \
    && TORCH_VERSION=$(grep -i '^torch==' /app/requirements.txt | cut -d'=' -f3) \
    && pip install --no-cache-dir setuptools==$SETUPTOOLS_VERSION \
    && pip install --no-cache-dir Cython==$CYTHON_VERSION \
    && pip install --no-cache-dir numpy==$NUMPY_VERSION \
    && pip install --no-cache-dir --prefer-binary pystan==$PYSTAN_VERSION \
    && pip install torch==$TORCH_VERSION --index-url https://download.pytorch.org/whl/cpu

# Install requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --timeout 1000

# Copy the crontab and setup the cron job
COPY settings/crontab /etc/cron.d/allora_worker_cron
RUN chmod 0644 /etc/cron.d/allora_worker_cron && crontab /etc/cron.d/allora_worker_cron

# Copy the allora-offchain-node binary from the first stage
COPY --from=allora_node /node/allora_offchain_node /usr/local/bin/allora_offchain_node
# Copy the allonode-data .env file
COPY ./allonode-data/.env /node/.env

# Copy the supervisor configuration file
COPY settings/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set permissions to the non-privileged user
RUN chown -R appuser:appgroup /app

# Expose the port that the app will run on
EXPOSE 8000

# Start supervisord to run both cron and uvicorn
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
