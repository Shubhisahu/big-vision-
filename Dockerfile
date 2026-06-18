FROM python:3.10-slim

# Create a non-root user (Required by Hugging Face Spaces security)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install
COPY --chown=user backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the rest of your codebase
COPY --chown=user . .


# Hugging Face Spaces requires the app to run on port 7860
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
