# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Builder
# Installs Python packages into an isolated prefix so only the built
# virtualenv is copied to the runtime image (keeps the final image lean).
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# System-level build dependencies:
#   build-essential  — gcc/g++ for packages with C extensions (e.g. Pillow)
#   libgl1-mesa-glx  — OpenGL shared library required by cv2 at *import* time
#   libglib2.0-0     — GLib shared library; cv2.VideoCapture depends on it
#   libsm6 libxrender1 libxext6 — X11 libs for OpenCV GUI code paths
#     (headless cv2 still links against them even if no GUI is opened)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create and activate an isolated venv — avoids polluting the system Python
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements first to leverage Docker layer caching:
# if requirements.txt is unchanged, `pip install` is skipped on rebuild.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runtime image
# Copies the pre-built venv and only the runtime system libraries.
# No compiler, no build tools, no cache — minimum attack surface.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    # Point Python to our venv without `source activate`
    PATH="/opt/venv/bin:$PATH" \
    # Disable .pyc bytecode files in the container (saves disk, no perf hit)
    PYTHONDONTWRITEBYTECODE=1 \
    # Force stdout/stderr to be unbuffered so Streamlit logs appear immediately
    PYTHONUNBUFFERED=1 \
    # Streamlit: disable the "email" nag prompt on first run
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

# Runtime-only shared libraries (same as builder; no compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — security best practice for containerised web services
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy virtualenv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source
COPY scanner.py app.py ./

# Transfer ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Health check: Streamlit exposes a /_stcore/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.enableCORS=false"]
